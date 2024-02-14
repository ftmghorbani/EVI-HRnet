import os
import math
import json
import logging
import datetime
import torch
import importlib
import warnings
import time
from time import localtime, strftime
import logging
import sys
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from evaluate import get_fid_score, get_i3d_activations, init_i3d_model, evaluate_video_error
from data_loader.data_loaders import FrameReader
from source.loss import AdversarialLoss

from data_loader.data_loaders import MaskedFrameDataLoader
import source.loss as module_loss
import source.metric as module_metric
from source.model import VideoInpaintingModel
import json



# Clean existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create log dir
LOG_DIR = 'logs'
created_time = strftime("%Y%m%d_%H%M%S", localtime())
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Set up handlers
LOGGING_LEVEL = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(f"{LOG_DIR}/{created_time}.log")
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

# Try to use colored formatter from coloredlogs
try:
    import coloredlogs
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    stream_handler.setFormatter(formatter)
except Exception as err:
    print(f"{err}")

handlers = [
    file_handler,
    stream_handler
]
logging.basicConfig(
    format=format_,
    level=LOGGING_LEVEL,
    handlers=handlers
)
logger = logging.getLogger(__name__)

class Logger:
    """
    Training process logger
    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory {dir_name} made")


ensure_dir = make_dirs

def save_frames_to_dir(frames, dirname):
    reader = FrameReader(dirname, read=False)
    reader.set_files(frames)
    reader.save_files(dirname)

class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = """
                    TensorboardX visualization is configured to use, but currently not installed on this machine.
                    Please install the package by 'pip install tensorboardx' command or turn off the option
                    in the 'config.json' file.
                """
                warnings.warn(message, UserWarning)
                # logger.warn()
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = [
            'add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram',
            'add_pr_curve', 'add_embedding'
        ]

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr



class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self, model, losses, metrics, optimizer_g,
        optimizer_d_s, optimizer_d_t,
        resume, n_gpu, epochs, save_freq, verbosity, pretrained_load_strict,
        monitor, monitor_mode, save_dir, name, log_dir, tensorboardX,
        train_logger=None,
        pretrained_path=None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(n_gpu)
        self.model = model.to(self.device)

        self.losses = losses
        self.metrics = metrics
        self.optimizer_g = optimizer_g
        self.optimizer_d_s = optimizer_d_s
        self.optimizer_d_t = optimizer_d_t

        self.epochs = epochs
        self.save_freq = save_freq
        self.verbosity = verbosity

        # Set pretrained_load_strict to False to load model without strict state name matching
        # It's useful when pretrained model without GAN but we want to use GAN for this time
        self.pretrained_load_strict = pretrained_load_strict

        self.train_logger = train_logger

        # configuration to monitor model performance and save best
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(save_dir, name, start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(log_dir, name, start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, tensorboardX)
        
        #update dict according to the config file
        config_dict = {
                'trainer': {
                    'monitor': self.monitor,
                    'monitor_mode': self.monitor_mode,
                    # Add other trainer-related configurations here
                },
                'visualization': {
                    'tensorboardX': tensorboardX,
                    # Add other visualization-related configurations here
                },
                # Add any other configuration sections here
            }

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config_dict, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
        elif pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # put model into DataParallel module only after the checkpoint is loaded
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = (f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                   f"but only {n_gpu} are available on this machine.")
            self.logger.warning(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            monitor_value = None
            if self.monitor_mode != 'off':
                try:
                    if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                            (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                        self.monitor_best = log[self.monitor]
                        best = True
                    monitor_value = log[self.monitor]

                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
                            + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.logger.warning(msg)
            if epoch % self.save_freq == 0 or best:
                self._save_checkpoint(epoch, save_best=best, monitor_value=monitor_value)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False, monitor_value=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
    

        # assure that we save the model state without DataParallel module
        if isinstance(self.model, torch.nn.DataParallel):
            # get the original state out from DataParallel module
            model_state = self.model.module.state_dict()
        else: 
            model_state = self.model.state_dict()
        state = {
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': model_state,
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d_s': self.optimizer_d_s.state_dict() if self.optimizer_d_s is not None else None,
            'optimizer_d_t': self.optimizer_d_t.state_dict() if self.optimizer_d_t is not None else None,
            'monitor_best': self.monitor_best,
        }

        best_str = '-best-so-far' if save_best else ''
        monitor_str = f'-{self.monitor}{monitor_value:.4f}' if monitor_value is not None else ''
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}{monitor_str}{best_str}.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        """ if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.') """
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if self.optimizer_d_s is not None:
            self.optimizer_d_s.load_state_dict(checkpoint['optimizer_d_s'])
        if self.optimizer_d_t is not None:
            self.optimizer_d_t.load_state_dict(checkpoint['optimizer_d_t'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def _load_pretrained(self, pretrained_path):
        self.logger.info(f"Loading pretrained checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        pretrained_state = checkpoint['state_dict']
        if self.pretrained_load_strict:
            self.model.load_state_dict(pretrained_state)
        else:
            current_state = self.model.state_dict()
            lack_modules = set([
                k.split('.')[0]
                for k in current_state.keys()
                if k not in pretrained_state.keys()
            ])
            self.logger.info("Allowing lack of submodules for pretrained model.")
            self.logger.info(f"Submodule(s) not in pretrained model but in current model: {lack_modules}")
            redundant_modules = set([
                k.split('.')[0]
                for k in pretrained_state.keys()
                if k not in current_state.keys()
            ])
            self.logger.info(f"Submodule(s) not in current model but in pretraired model: {set(redundant_modules)}")

            # used_pretrained_state = {k: v for k, v in pretrained_state.items() if k in current_state}
            used_pretrained_state = {}
            prefixs = [
                'generator.coarse_net.upsample_module.',
                'generator.coarse_net.downsample_module.',
            ]
            for k, v in pretrained_state.items():
                if k in current_state:
                    used_pretrained_state[k] = v
                else:
                    # Backward compatible
                    for prefix in prefixs:
                        new_key = prefix + k
                        if new_key in current_state:
                            self.logger.warning(f"Load key to new model: {k} -> {new_key}")
                            used_pretrained_state[new_key] = v
            current_state.update(used_pretrained_state)
            self.model.load_state_dict(current_state)
        self.logger.info("Pretrained checkpoint loaded")





class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self, model, losses, metrics,
        optimizer_g, optimizer_d_s, optimizer_d_t, resume, n_gpu, epochs, save_freq, verbosity, pretrained_load_strict,
        monitor, monitor_mode, save_dir, name, log_dir, tensorboardX, data_loader,
        valid_data_loader=None, lr_scheduler=None, train_logger=None, learn_mask=True,
        test_data_loader=None, pretrained_path=None, log_step=None, loss_gan_s_w=None,
        loss_gan_t_w=None, evaluate_score=True, store_gated_values=False, printlog=False,
    ):
        super().__init__(
            model, losses, metrics, optimizer_g,
            optimizer_d_s, optimizer_d_t, resume, n_gpu, epochs, save_freq, verbosity, pretrained_load_strict,
            monitor, monitor_mode, save_dir, name, log_dir, tensorboardX, train_logger,
            pretrained_path
        )

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step
        self.loss_gan_s_w = loss_gan_s_w
        self.loss_gan_t_w = loss_gan_t_w
        # theirs: lsgan, ours: wgan
        self.adv_loss_fn = AdversarialLoss(type='lsgan')
        self.evaluate_score = evaluate_score
        self.store_gated_values = store_gated_values
        self.printlog = printlog

        if self.test_data_loader is not None:
            self.toPILImage = ToPILImage()
            self.test_output_root_dir = os.path.join(self.checkpoint_dir, 'test_outputs')
        init_i3d_model()

    def _store_gated_values(self, out_dir):
        from source.blocks import GatedConv, GatedDeconv

        def save_target(child, out_subdir):
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            if isinstance(child, GatedConv):
                target = child.gated_values[0]
            elif isinstance(child, GatedDeconv):
                target = child.conv.gated_values[0]
            else:
                raise ValueError('should be gated conv or gated deconv')
            target = target.transpose(0, 1)
            for t in range(target.shape[0]):
                for c in range(target.shape[1]):
                    out_file = os.path.join(out_subdir, f'time{t:03d}_channel{c:04d}.png')
                    self.toPILImage(target[t, c: c + 1]).save(out_file)

        for key, child in self.model.generator.coarse_net.upsample_module.named_children():
            out_subdir = os.path.join(out_dir, f'upsample_{key}')
            save_target(child, out_subdir)
        for key, child in self.model.generator.coarse_net.downsample_module.named_children():
            out_subdir = os.path.join(out_dir, f'downsample_{key}')
            save_target(child, out_subdir)

    def _evaluate_data_loader(self, epoch=None, output_root_dir=None, data_loader=None, name='test'):
        total_length = 0
        total_error = 0
        total_psnr = 0
        total_ssim = 0
        total_p_dist = 0

        if output_root_dir is None:
            output_root_dir = self.test_output_root_dir
        if epoch is not None:
            output_root_dir = os.path.join(output_root_dir, f"epoch_{epoch}")
        output_root_dir = os.path.join(output_root_dir, name)

        output_i3d_activations = []
        real_i3d_activations = []
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data_input, model_output = self._process_data(data)
                inputs, outputs, targets, masks, ref_frames = self._unpack_data(data_input, model_output)
                if self.store_gated_values:
                    out_dir = os.path.join(output_root_dir, 'gated_values', f'input_{batch_idx:04}')
                    self._store_gated_values(out_dir)
                outputs = outputs.clamp(0, 1)

                if self.evaluate_score:
                    # get i3d activation
                    output_i3d_activations.append(get_i3d_activations(outputs).cpu().numpy())
                    real_i3d_activations.append(get_i3d_activations(targets).cpu().numpy())

                assert len(outputs) == 1  # Batch size = 1 for testing
                inputs = inputs[0]
                outputs = outputs[0].cpu()
                targets = targets[0].cpu()
                masks = masks[0].cpu()

                if epoch is not None and epoch == 1:
                    # Save inputs to output_dir
                    output_dir = os.path.join(output_root_dir, 'inputs', f"input_{batch_idx:04}")
                    self.logger.debug(f"Saving batch {batch_idx} input to {output_dir}")
                    save_frames_to_dir([self.toPILImage(t) for t in inputs.cpu()], output_dir)

                    # to save targets:
                    output_dir_targets = os.path.join(output_root_dir, 'targets', f"target_{batch_idx:04}")
                    self.logger.debug(f"Saving batch {batch_idx} targets to {output_dir_targets}")
                    save_frames_to_dir([self.toPILImage(t) for t in targets], output_dir_targets)


                if epoch is not None and epoch % 5 == 0:
                    # Save test results to output_dir
                    output_dir = os.path.join(output_root_dir, f"result_{batch_idx:04}")
                    self.logger.debug(f"Saving batch {batch_idx} to {output_dir}")
                    save_frames_to_dir([self.toPILImage(t) for t in outputs], output_dir)

                    # to save targets:
                    output_dir_targets = os.path.join(output_root_dir, 'targets', f"target_{batch_idx:04}")
                    self.logger.debug(f"Saving batch {batch_idx} targets to {output_dir_targets}")
                    save_frames_to_dir([self.toPILImage(t) for t in targets], output_dir_targets)

                if self.evaluate_score:
                    # Evaluate scores
                    error, psnr_value, ssim_value, p_dist, length = \
                        self._evaluate_test_video(outputs, targets, masks)
                    
                    total_error += error
                    total_ssim += ssim_value
                    total_psnr += psnr_value
                    total_p_dist += p_dist
                    total_length += length

        if self.evaluate_score:
            output_i3d_activations = np.concatenate(output_i3d_activations, axis=0)
            real_i3d_activations = np.concatenate(real_i3d_activations, axis=0)
            fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        else:
            fid_score = 0
            total_p_dist = [0]
            total_length = 1
        total_p_dist = total_p_dist[0]

        if epoch is not None:
            self.writer.set_step(epoch, name)
            self._write_images(
                inputs, ref_frames, outputs, targets, masks,
                model_output=model_output, data_input=data_input
            )

            self.writer.add_scalar('test_mse', total_error / total_length)
            self.writer.add_scalar('test_ssim', total_ssim / total_length)
            self.writer.add_scalar('test_psnr', total_psnr / total_length)
            self.writer.add_scalar('test_p_dist', total_p_dist / total_length)
            self.writer.add_scalar('test_fid_score', fid_score)
        return total_error, total_ssim, total_psnr, total_p_dist, total_length, fid_score

    def _write_images(
            self, inputs, ref_frames, outputs, targets, masks, output_edges=None,
            target_edges=None, model_output=None, data_input=None
    ):
        self.writer.add_image('input', make_grid(inputs.cpu(), nrow=3, normalize=False, data_format='CHW'))
        self.writer.add_image('loss_mask', make_grid(masks.cpu(), nrow=3, normalize=False))
        self.writer.add_image(
            'output', make_grid(outputs.clamp(0, 1).cpu(), nrow=3, normalize=False,))
        self.writer.add_image('gt', make_grid(targets.cpu(), nrow=3, normalize=False))
        self.writer.add_image('diff', make_grid(targets.cpu() - outputs.cpu(), nrow=3, normalize=True))
        self.writer.add_image('IO_diff', make_grid(inputs.cpu() - outputs.cpu(), nrow=3, normalize=True))
        self.writer.add_image('input_frames', make_grid(inputs.cpu(), nrow=3, normalize=False))
        #self.writer.add_image('reference_frames', make_grid(ref_frames.cpu(), nrow=3, normalize=False, data_format='CHW'))

        try:
            output_edges = self.losses['loss_edge'][0].current_output_edges
            target_edges = self.losses['loss_edge'][0].current_target_edges
            self.writer.add_image('output_edge', make_grid(output_edges[0].cpu(), nrow=3, normalize=True))
            self.writer.add_image('target_edge', make_grid(target_edges[0].cpu(), nrow=3, normalize=True))
        except Exception:
            pass
        try:
            guidances = data_input['guidances']
            self.writer.add_image('guidances', make_grid(guidances[0].cpu(), nrow=3, normalize=True))
        except Exception:
            pass

        if model_output is not None:
            if 'imcomplete_video' in model_output.keys():
                self.writer.add_image('imcomplete_video', make_grid(
                    model_output['imcomplete_video'][0].transpose(0, 1).cpu(), nrow=3, normalize=False))

    def _evaluate_test_video(self, output, gt_frames, masks):
        gt_images = [self.toPILImage(gt) for gt in gt_frames]
        result_images = [self.toPILImage(result) for result in output]
        mask_images = [self.toPILImage(mask / 255) for mask in masks]
        return evaluate_video_error(
            result_images, gt_images, mask_images,
            printlog=self.printlog
        )

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _get_gan_loss(self, outputs, target, masks, discriminator, w, guidances=None, is_disc=None):
        if w <= 0:
            return torch.Tensor([0]).to(self.device)
        scores = self.model.forward(outputs, masks, guidances, model=discriminator)
        gan_loss = self.adv_loss_fn(scores, target, is_disc)
        return gan_loss

    def _get_grad_mean_magnitude(self, output, optimizer):
        """
        Get mean magitude (absolute value) of gradient of output w.r.t params in the optimizer.
        This function is used to get a simple understanding over the impact of a loss.

        :output: usually the loss you want to compute gradient w.r.t params
        :optimizer: the optimizer who contains the parameters you care

        Note:
            This function will reset the gradient stored in paramerter, so please
            use it before <your loss>.backward()

        Example:
            > grad_magnitude = self._get_grad_mean_magnitude(
                  loss_recon * self.loss_recon_w, self.optimizer_g))
            > print(grad_magnitude)
        """
        optimizer.zero_grad()
        output.backward(retain_graph=True)
        all_grad = []
        for group in optimizer.param_groups:
            for p in group['params']:
                all_grad.append(p.grad.view(-1))
        value = torch.cat(all_grad).abs().mean().item()
        optimizer.zero_grad()
        return value

    

    def _process_data(self, data):
        inputs = data["input_tensors"].to(self.device)
        masks = data["mask_tensors"].to(self.device)
        targets = data["gt_tensors"].to(self.device)
        # guidances = self._get_edge_guidances(targets).to(self.device) if 'edge' in data['guidance'] else None
        guidances = data["guidances"].to(self.device) if len(data["guidances"]) > 0 else None
        ref_frames = data["reference_tensors"].to(self.device) if "reference_tensors" in data else None
        data_input = {
            "inputs": inputs,
            "masks": masks,
            "targets": targets,
            "guidances": guidances,
            "ref_frames": ref_frames
        }

        model_output = self.model(inputs, masks, ref_frames, guidances)
        return data_input, model_output

    def _unpack_data(self, data_input, model_output):
        # inputs, outputs, targets, masks = self._unpack_data(data_input, model_output)
        return (
            data_input['inputs'],
            model_output['outputs'] if 'refined_outputs' not in model_output.keys()
            else model_output['refined_outputs'],
            data_input['targets'],
            data_input['masks'],
            data_input['ref_frames']
        )

    def _get_non_gan_loss(self, data_input, model_output):
        # Compute and write all non-GAN losses to tensorboard by for loop
        losses = []
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight > 0.0:
                loss = loss_instance(data_input, model_output)
                self.writer.add_scalar(f'{loss_name}', loss.item())
                loss *= loss_weight
                losses.append(loss)
        loss = sum(losses)
        return loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        epoch_start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()

            # Set writer
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            data_input, model_output = self._process_data(data)
            inputs, outputs, targets, masks, ref_frames= self._unpack_data(data_input, model_output)

            # Train G
            non_gan_loss = self._get_non_gan_loss(data_input, model_output)

            loss_gan_s = self._get_gan_loss(
                outputs, 1, masks, discriminator='D_s', w=self.loss_gan_s_w, is_disc=False)
            loss_gan_t = self._get_gan_loss(
                outputs, 1, masks, discriminator='D_t', w=self.loss_gan_t_w, is_disc=False)

            loss_total = (
                non_gan_loss
                + loss_gan_s * self.loss_gan_s_w
                + loss_gan_t * self.loss_gan_t_w
            )

            self.optimizer_g.zero_grad()

            # Uncomment these lines to see the gradient
            # grad_recon = self._get_grad_mean_magnitude(loss_recon, self.optimizer_g)
            # grad_vgg = self._get_grad_mean_magnitude(loss_vgg, self.optimizer_g)
            # grad_gan_s = self._get_grad_mean_magnitude(loss_gan_s, self.optimizer_g)
            # grad_gan_t = self._get_grad_mean_magnitude(loss_gan_t, self.optimizer_g)
            # self.logger.info(f"Grad: recon {grad_recon} vgg {grad_vgg} gan_s {grad_gan_s} gan_t {grad_gan_t}")

            loss_total.backward()
            self.optimizer_g.step()

            # Train spatial and temporal discriminators
            for d in ['s', 't']:
                weight = getattr(self, f'loss_gan_{d}_w')
                optimizer = getattr(self, f'optimizer_d_{d}')

                if weight > 0:
                    optimizer.zero_grad()
                    loss_d = (
                        self._get_gan_loss(
                            targets, 1, masks, discriminator=f'D_{d}', w=weight, is_disc=True)
                        + self._get_gan_loss(
                            outputs.detach(), 0, masks, discriminator=f'D_{d}', w=weight, is_disc=True)
                    ) / 2
                    loss_d.backward()
                    optimizer.step()

                    self.writer.add_scalar(f'loss_d_{d}', loss_d.item())

            self.writer.add_scalar('loss_total', loss_total.item())
            self.writer.add_scalar('loss_gan_s', loss_gan_s.item())
            self.writer.add_scalar('loss_gan_t', loss_gan_t.item())

            with torch.no_grad():
                total_loss += loss_total.item()
                total_metrics += self._eval_metrics(outputs, targets)

            if self.verbosity >= 2 and \
                    (batch_idx % self.log_step == 0 and epoch < 30) or \
                    batch_idx == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
                    f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
                    f'loss_total: {loss_total.item():.3f}, '
                    f'BT: {time.time() - batch_start_time:.2f}s'
                )

                self._write_images(inputs[0], ref_frames[0], outputs[0], targets[0], masks[0],
                                   model_output=model_output, data_input=data_input)

        log = {
            'epoch_time': time.time() - epoch_start_time,
            'loss_total': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.test_data_loader is not None:
            log = self.evaluate_test_set(epoch=epoch, log=log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            #self.lr_scheduler.step(log['test_mse'])

        return log

    def evaluate_test_set(self, output_root_dir=None, epoch=None, log=None):
        # Insert breakpoint when Nan
        self.model.eval()
        if isinstance(self.test_data_loader, list):
            test_data_loaders = self.test_data_loader
        else:
            test_data_loaders = [self.test_data_loader]
        try:
            for i, data_loader in enumerate(test_data_loaders):
                total_error, total_ssim, total_psnr, total_p_dist, total_length, fid_score = \
                    self._evaluate_data_loader(data_loader=data_loader,
                                               output_root_dir=output_root_dir, epoch=epoch)

                
                if self.printlog:
                    self.logger.info(f'test_mse: {total_error / total_length}')
                    self.logger.info(f'test_ssim: {total_ssim / total_length}')
                    self.logger.info(f'test_psnr: {total_psnr / total_length}')
                    self.logger.info(f'test_p_dist: {total_p_dist / total_length}')
                    self.logger.info(f'test_fid_score: {fid_score}\n')
        except Exception as err:
            self.logger.error(err, exc_info=True)
            breakpoint()  # NOQA
        if log is not None:
            log['test_mse'] = total_error / total_length
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        self.logger.info(f"Doing {epoch} validation ..")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                if epoch == 1 and batch_idx > 5:
                    continue
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                data_input, model_output = self._process_data(data)
                inputs, outputs, targets, masks, ref_frames = self._unpack_data(data_input, model_output)

                loss_total = self._get_non_gan_loss(data_input, model_output)

                self.writer.add_scalar('loss_total', loss_total.item())
                total_val_loss += loss_total.item()
                total_val_metrics += self._eval_metrics(outputs, targets)

                if batch_idx % self.log_step == 0:
                    self._write_images(
                        inputs[0], ref_frames[0], outputs[0], targets[0], masks[0],
                        model_output=model_output, data_input=data_input
                    )

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist(),
        }


#torch.cuda.empty_cache()
    
# Read JSON config file
with open('config.json', 'r') as f:
    config = json.load(f)

model_config = config['model_config']
model = VideoInpaintingModel(model_config)

model.summary()

metrics = [module_metric.L2_loss]

g_params = []
d_s_params = []
d_t_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'temporal_discriminator' in name:
        d_t_params.append(param)
    elif 'spatial_discriminator' in name:
        d_s_params.append(param)
    else:
        g_params.append(param)


# Replace these lines with your desired optimizer settings
optimizer_type = torch.optim.Adam
optimizer_args = config['optimizer_args']

optimizer_g = optimizer_type(g_params, **optimizer_args)
if hasattr(model, 'spatial_discriminator'):
    optimizer_d_s = optimizer_type(d_s_params, **optimizer_args)
else:
    optimizer_d_s = None
if hasattr(model, 'temporal_discriminator'):
    optimizer_d_t = optimizer_type(d_t_params, **optimizer_args)
else:
    optimizer_d_t = None


# Now you can use the config dictionary to access your configurations
losses = {
    "StyleLoss": (module_loss.StyleLoss(), config['losses']["StyleLoss"][0]),
    "VGGLoss": (module_loss.VGGLoss(), config['losses']["VGGLoss"][0]),
    "ExpressionLoss": (module_loss.ExpressionLoss(), config['losses']["ExpressionLoss"][0]),
    "ReconLoss": (module_loss.ReconLoss(masked=False), config['losses']["ReconLoss"][0]),
    "ReconLoss_masked": (module_loss.ReconLoss(masked=True), config['losses']["ReconLoss"][1]),
    "CompleteFramesReconLoss": (module_loss.CompleteFramesReconLoss(), config['losses']["CompleteFramesReconLoss"][0]),
}


optimizer_args = config['optimizer_args']
epochs = config['epochs']
n_gpu = config['n_gpu']
save_freq = config['save_freq']
verbosity = config['verbosity']
pretrained_load_strict = config['pretrained_load_strict']
monitor = config['monitor']
monitor_mode = config['monitor_mode']
save_dir = config['save_dir']
name = config['name']
log_dir = config['log_dir']
tensorboardX = config['tensorboardX']
lr_scheduler_config = config['lr_scheduler']
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=lr_scheduler_config['step_size'], gamma=lr_scheduler_config['gamma'])
pretrained_path = config['pretrained_path']
root_videos_dir = config['root_videos_dir']
root_masks_dir = config['root_masks_dir']
root_outputs_dir = config['root_outputs_dir']
root_reference_frames_dir = config['root_reference_frames_dir']
batch_size = config['batch_size']
shuffle = config['shuffle']
validation_split = config['validation_split']
num_workers = config['num_workers']
dataset_args = config['dataset_args']
root_videos_dir_T = config['root_videos_dir_T']
root_masks_dir_T = config['root_masks_dir_T']
root_reference_frames_dir_T = config['root_reference_frames_dir_T']
dataset_args_T = config['dataset_args_T']
shuffle_T = config['shuffle']
log_step = config['log_step']
loss_gan_s_w = config['loss_gan_s_w']
loss_gan_t_w = config['loss_gan_t_w']
resume = config['resume']
evaluate_score = config['evaluate_score']
store_gated_values = config['store_gated_values']
printlog = config['printlog']
valid_data_loader = config['valid_data_loader']

data_loader = MaskedFrameDataLoader(
    root_videos_dir, root_masks_dir, root_outputs_dir, root_reference_frames_dir,
    dataset_args,
    batch_size, shuffle, validation_split,
    num_workers
)
data_loader_T = MaskedFrameDataLoader(
    root_videos_dir_T, root_masks_dir_T, root_outputs_dir, root_reference_frames_dir_T,
    dataset_args_T,
    batch_size, shuffle_T, validation_split, num_workers)

# Run your training process
trainer = Trainer(
    model, losses, metrics, optimizer_g, optimizer_d_s, optimizer_d_t,
    resume=resume,
    n_gpu=n_gpu,
    epochs=epochs,
    save_freq=save_freq,
    verbosity=verbosity,
    pretrained_load_strict=pretrained_load_strict,
    monitor=monitor,
    monitor_mode=monitor_mode,
    save_dir=save_dir,
    name=name,
    log_dir=log_dir,
    tensorboardX=tensorboardX,
    data_loader=data_loader,
    valid_data_loader=valid_data_loader,
    lr_scheduler=lr_scheduler,
    train_logger=Logger(),
    test_data_loader=data_loader_T,
    pretrained_path=pretrained_path,
    log_step=log_step,
    loss_gan_s_w=loss_gan_s_w,
    loss_gan_t_w=loss_gan_t_w,
    evaluate_score=evaluate_score,
    store_gated_values=store_gated_values,
    printlog=printlog,
)

output_root_dir = config['output_root_dir']

if output_root_dir is not None:
        make_dirs(output_root_dir)
        trainer.printlog = True
        trainer.evaluate_test_set(
            output_root_dir=output_root_dir, epoch=0)
else:
        trainer.train()

