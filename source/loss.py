import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # NOQA

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
#consider each later, add or remove
from source.vgg import Vgg16
import torch.nn.functional as F
from hsemotion.facial_emotions import HSEmotionRecognizer


device = torch.device("cuda")
vgg = Vgg16(requires_grad=False).to(device)

class ReconLoss(nn.Module):
    def __init__(self, reduction='mean', masked=False):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        self.masked = masked

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        if self.masked:
            masks = data_input['masks']
            return self.loss_fn(outputs * (1 - masks), targets * (1 - masks))
        else:
            return self.loss_fn(outputs, targets)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def vgg_loss(self, output, target):
        output_feature = vgg(output)
        target_feature = vgg(target)
        loss = (
            self.l1_loss(output_feature.relu2_2, target_feature.relu2_2)
            + self.l1_loss(output_feature.relu3_3, target_feature.relu3_3)
            + self.l1_loss(output_feature.relu4_3, target_feature.relu4_3)
        )
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.vgg_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss


class StyleLoss(nn.Module):
    def __init__(self, original_channel_norm=True):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.original_channel_norm = original_channel_norm

    # From https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    # Implement "Image Inpainting for Irregular Holes Using Partial Convolutions", Liu et al., 2018
    def style_loss(self, output, target):
        output_features = vgg(output)
        target_features = vgg(target)
        layers = ['relu2_2', 'relu3_3', 'relu4_3']  # n_channel: 128 (=2 ** 7), 256 (=2 ** 8), 512 (=2 ** 9)
        loss = 0
        for i, layer in enumerate(layers):
            output_feature = getattr(output_features, layer)
            target_feature = getattr(target_features, layer)
            B, C_P, H, W = output_feature.shape
            output_gram_matrix = self.gram_matrix(output_feature)
            target_gram_matrix = self.gram_matrix(target_feature)
            if self.original_channel_norm:
                C_P_square_divider = 2 ** (i + 1)  # original design (avoid too small loss)
            else:
                C_P_square_divider = C_P ** 2
                assert C_P == 128 * 2 ** i
            loss += self.l1_loss(output_gram_matrix, target_gram_matrix) / C_P_square_divider
        return loss

    def forward(self, data_input, model_output):
        targets = data_input['targets']
        outputs = model_output['outputs']

        # Note: It can be batch-lized
        mean_image_loss = []
        for frame_idx in range(targets.size(1)):
            mean_image_loss.append(
                self.style_loss(outputs[:, frame_idx], targets[:, frame_idx])
            )

        mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        return mean_image_loss



class L1LossMaskedMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, mask):
        masked = 1 - mask
        l1_sum = self.l1(x * masked, y * masked)
        return l1_sum / torch.sum(masked)


class L2LossMaskedMean(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.l2 = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, mask):
        masked = 1 - mask
        l2_sum = self.l2(x * masked, y * masked)
        return l2_sum / torch.sum(masked)


class ImcompleteVideoReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        imcomplete_video = model_output['imcomplete_video']
        targets = data_input['targets']
        down_sampled_targets = nn.functional.interpolate(
            targets.transpose(1, 2), scale_factor=[1, 0.5, 0.5])

        masks = data_input['masks']
        down_sampled_masks = nn.functional.interpolate(
            masks.transpose(1, 2), scale_factor=[1, 0.5, 0.5])
        return self.loss_fn(
            imcomplete_video, down_sampled_targets,
            down_sampled_masks
        )


class CompleteFramesReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = L1LossMaskedMean()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        masks = data_input['masks']
        return self.loss_fn(outputs, targets, masks)



# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='wgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1 | wgan
        theirs: lsgan, ours: wgan
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).to(device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label).to(device))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        elif type == 'l1':
            self.criterion = nn.L1Loss()

        elif type == 'wgan':
            self.criterion = self._wasserstein_loss

    def _wasserstein_loss(self, outputs, labels):
        return torch.mean(outputs * labels)

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss



class ExpressionLoss(nn.Module):
    def __init__(self, expression_model='enet_b0_8_best_afew'):
        super().__init__()
        self.expression_model = HSEmotionRecognizer('enet_b0_8_best_afew')
        self.to_pil = ToPILImage(mode='RGB')
        self.to_tensor = ToTensor()

    def forward(self, data_input, model_output):
        outputs = model_output['outputs']
        targets = data_input['targets']
        
        batch_size, num_frames, channels, height, width = targets.shape

        input_scores = []
        output_scores = []

        for i in range(batch_size):
            for j in range(num_frames):
                input_frame = targets[i, j].cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
                output_frame = outputs[i, j].cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)

                _, input_score = self.expression_model.predict_emotions(input_frame)
                _, output_score = self.expression_model.predict_emotions(output_frame)

                input_scores.append(input_score)
                output_scores.append(output_score)

        input_scores_array = np.array(input_scores)
        output_scores_array = np.array(output_scores)

        input_scores_tensor = torch.tensor(input_scores_array)
        output_scores_tensor = torch.tensor(output_scores_array)

        # Compute the mean squared error between the input and output expression scores
        mae_loss_FER = F.l1_loss(input_scores_tensor, output_scores_tensor)
        #mse_loss_FER = F.mse_loss(input_scores_tensor, output_scores_tensor)

        return mae_loss_FER
