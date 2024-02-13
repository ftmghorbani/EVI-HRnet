import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import functools
import operator

from source.tsm_utils import tsm

########################
# Convolutional Blocks #
########################



class NN3Dby2D(object):
    '''
    Use these inner classes to mimic 3D operation by using 2D operation frame by frame.
    '''
    class Base(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, xs):
            dim_length = len(xs.shape)
            if dim_length == 5:  # [batch_size, channels, video_len, w, h]
                # Unbind the video data to a tuple of frames
                xs = torch.unbind(xs, dim=2)
                # Process them frame by frame using 2d layer
                xs = torch.stack([self.layer(x) for x in xs], dim=2)
            elif dim_length == 4:  # [batch_size, channels, w, h]
                # keep the 2D ability when the data is not batched videoes but batched frames
                xs = self.layer(xs)
            return xs

    class Conv3d(Base):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride, padding, dilation, groups, bias):
            super().__init__()
            # take off the kernel/stride/padding/dilation setting for the temporal axis
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[1:]
            if isinstance(stride, tuple):
                stride = stride[1:]
            if isinstance(padding, tuple):
                padding = padding[1:]
            if isinstance(dilation, tuple):
                dilation = dilation[1:]
            self.layer = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias
            )

            # let the spectral norm function get its conv weights
            self.weight = self.layer.weight
            # let partial convolution get its conv bias
            self.bias = self.layer.bias
            self.__class__.__name__ = "Conv3dBy2D"

    class BatchNorm3d(Base):
        def __init__(self, out_channels):
            super().__init__()
            self.layer = nn.BatchNorm2d(out_channels)

    class InstanceNorm3d(Base):
        def __init__(self, out_channels, track_running_stats=True):
            super().__init__()
            self.layer = nn.InstanceNorm2d(out_channels, track_running_stats=track_running_stats)


class NN3Dby2DTSM(NN3Dby2D):

    class Conv3d(NN3Dby2D.Conv3d):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride, padding, dilation, groups, bias):
            super().__init__(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias
            )
            self.__class__.__name__ = "Conv3dBy2DTSM"

        def forward(self, xs):
            # identity = xs
            B, C, L, H, W = xs.shape
            # Unbind the video data to a tuple of frames
            xs_tsm = tsm(xs.transpose(1, 2), L, 'zero').contiguous()
            out = self.layer(xs_tsm.view(B * L, C, H, W))
            _, C_, H_, W_ = out.shape
            return out.view(B, L, C_, H_, W_).transpose(1, 2)

            return out


class VanillaConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):

        super().__init__()
        if conv_by == '2d':
            self.module = NN3Dby2D
        elif conv_by == '2dtsm':
            self.module = NN3Dby2DTSM
        elif conv_by == '3d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')

        self.padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2) if padding == -1 else padding
        self.featureConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = self.module.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = self.module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_by='3d'
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class GatedConv(VanillaConv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by
        )
        if conv_by == '2dtsm':
            self.module = NN3Dby2D
        self.gatingConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
        self.sigmoid = nn.Sigmoid()
        self.store_gated_values = False

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        out = self.sigmoid(mask)
        if self.store_gated_values:
            self.gated_values = out.detach().cpu()
        return out

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(VanillaDeconv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_by='3d'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, scale_factor, conv_by
        )
        self.conv = GatedConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)

