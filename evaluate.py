import os
import sys
import argparse
import urllib.request
from glob import glob
import math
import numpy as np
import torch
from torchvision import transforms
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from libs import PerceptualSimilarity

from data_loader.data_loaders import FrameReader, MaskReader, Stack, ToTorchFormatTensor
from data_loader.data_loaders import make_dirs
from source.i3d import InceptionI3d

def mean_squared_error(A, B):
    return np.square(np.subtract(A, B)).mean()



def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


MIN_PERCEPTUAL_SIZE = 64
i3d_model = None
dm_model = None

def init_i3d_model():
    global i3d_model
    if i3d_model is not None:
        return

    print("Loading I3D model for FID score ..")
    i3d_model_weight = '/home/ghorbani/MyCode_VidInp/libs/model_weights/rgb_imagenet.pt'
    if not os.path.exists(i3d_model_weight):
        make_dirs(os.path.dirname(i3d_model_weight))
        urllib.request.urlretrieve('https://github.com/piergiaj/pytorch-i3d/'
                                   'raw/master/models/rgb_imagenet.pt', i3d_model_weight)
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.to(torch.device('cuda:0'))


def init_dm_model():
    global dm_model
    if dm_model is not None:
        return

    print("Loading PerceptualSimilarity model ..")
    dm_model = PerceptualSimilarity.dist_model.DistModel()
    dm_model.initialize(model='net-lin', net='alex', use_gpu=True)

# def get_perceptual_distance(gt, result):
def evaluate_image(gt, result):
    gt = np.array(gt)
    result = np.array(result)
    mse = mean_squared_error(gt / 255, result / 255)
    psnr_value = psnr(gt / 255, result / 255)
    ssim_value = ssim(gt / 255, result / 255, channel_axis=-1)

    gt_tensor = PerceptualSimilarity.util.util.im2tensor(gt[..., :3])
    result_tensor = PerceptualSimilarity.util.util.im2tensor(result[..., :3])

    init_dm_model()
    p_dist = dm_model.forward(gt_tensor, result_tensor)
    return mse, ssim_value, psnr_value, p_dist


def evaluate_video(
    result_video_dir: str,
    gt_video_dir: str,
    video_mask_dir: str):
    result_frame_reader = FrameReader(result_video_dir).files
    gt_frame_reader = FrameReader(gt_video_dir).files
    if os.path.exists(video_mask_dir):
        masks = MaskReader(video_mask_dir)[:len(gt_frame_reader)]
    else:
        raise IOError(f"{video_mask_dir} not exists")

    if len(masks) != len(result_frame_reader):
        print("Size mismatch")
    return evaluate_video_error(result_frame_reader, gt_frame_reader, masks,
                                )


def evaluate_video_error(
    result_images, gt_images, masks,
    printlog=True
):
    total_error = 0
    total_psnr = 0
    total_ssim = 0
    total_p_dist = 0
    for i, (result, gt, mask) in enumerate(
        zip(result_images, gt_images, masks)
    ):
        # mask = np.expand_dims(mask, 2)
        mse, ssim_value, psnr_value, p_dist = evaluate_image(gt, result)

        total_error += mse
        total_ssim += ssim_value
        total_psnr += psnr_value
        total_p_dist += p_dist
        """ print(
            f"Frame {i}: MSE {mse} PSNR {psnr_value} SSIM {ssim_value} "
            f"Percep. Dist. {p_dist}"
        ) """



    if printlog:
        print(f"Avg MSE: {total_error / len(result_images)}")
        print(f"Avg PSNR: {total_psnr / len(result_images)}")
        print(f"Avg SSIM: {total_ssim / len(result_images)}")
        print(
            f"Avg Perce. Dist.: {total_p_dist / len(result_images)}")
    if total_error == 0:
        raise IOError("Error = 0")
    return (
        total_error, total_psnr, total_ssim, total_p_dist,
        len(result_images)
    )

def get_everything_under(root_dir, pattern='*', only_dirs=False, only_files=False):
    assert not(only_dirs and only_files), 'You will get nothnig '\
        'when "only_dirs" and "only_files" are both set to True'
    everything = sorted(glob(os.path.join(root_dir, pattern)))
    if only_dirs:
        everything = [f for f in everything if os.path.isdir(f)]
    if only_files:
        everything = [f for f in everything if os.path.isfile(f)]

    return everything

def evaluate_all_videos(
    root_gt_dir, root_result_dir,
    root_mask_dir, test_num, result_postfix="",
):

    total_error = 0
    total_psnr = 0
    total_ssim = 0
    total_p_dist = 0
    total_length = 0

    result_dirs = get_everything_under(root_result_dir, only_dirs=True)[:test_num]
    gt_dirs = get_everything_under(root_gt_dir, only_dirs=True)[:test_num]
    mask_dirs = get_everything_under(root_mask_dir, only_dirs=True)[:test_num]

    for i, (result_dir, gt_dir) in enumerate(zip(result_dirs, gt_dirs)):
        result_dir = os.path.join(result_dir, result_postfix)
        mask_dir = mask_dirs[i]
        print(f"Processing {result_dir}, mask {mask_dir}, gt {gt_dir}")
        error, psnr_value, ssim_value, p_dist, length = \
            evaluate_video(result_dir, gt_dir, mask_dir)
        
        total_error += error
        total_ssim += ssim_value
        total_psnr += psnr_value
        total_p_dist += p_dist
        total_length += length

    avg_mse_error = total_error / total_length
    avg_ssim = total_ssim / total_length
    avg_psnr = total_psnr / total_length
    avg_p_dist = (total_p_dist / total_length)[0]
    print(f"Total avg error {avg_mse_error:.5f}")
    print(f"Total avg ssim {avg_ssim:.4f}")
    print(f"Total avg PSNR {avg_psnr:.2f}")
    print(f"Total avg Perceptual distance {avg_p_dist:.4f}")
    print(f"Total length {total_length}")
    print(f"Video num: {len(result_dirs)}")
    return (avg_mse_error, avg_ssim, avg_psnr, avg_p_dist)


def evaluate_fid_score(root_gt_dir, root_result_dir, result_postfix):
    to_tensors = transforms.Compose([
        Stack(),
        ToTorchFormatTensor(),
    ])
    result_dirs = get_everything_under(root_result_dir, only_dirs=True)
    gt_dirs = get_everything_under(root_gt_dir, only_dirs=True)

    output_i3d_activations = []
    real_i3d_activations = []

    with torch.no_grad():
        for i, (result_dir, gt_dir) in enumerate(zip(result_dirs, gt_dirs)):
            if i % 20 == 0:
                print(f"Getting {i}th i3d activations")
            result_dir = os.path.join(result_dir, result_postfix)
            result_frame_reader = FrameReader(result_dir).files
            gt_frame_reader = FrameReader(gt_dir).files

            # Unsqueeze batch dimension
            outputs = to_tensors(result_frame_reader).unsqueeze(0).to(torch.device('cuda:0'))
            targets = to_tensors(gt_frame_reader).unsqueeze(0).to(torch.device('cuda:0'))
            # get i3d activation
            output_i3d_activations.append(get_i3d_activations(outputs).cpu().numpy())
            real_i3d_activations.append(get_i3d_activations(targets).cpu().numpy())
        # concat and evaluate fid score
        output_i3d_activations = np.concatenate(output_i3d_activations, axis=0)
        real_i3d_activations = np.concatenate(real_i3d_activations, axis=0)
        fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    print(f"Video num: {len(result_dirs)}")
    print(f"FID score: {fid_score}")
    return fid_score


def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    """
    init_i3d_model()
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat


def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)


# code from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)