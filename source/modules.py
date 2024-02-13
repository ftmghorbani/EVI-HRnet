import torch
import torch.nn as nn

from torch.nn import functional as F
from source.dcn import DeformConv3d

from source.blocks import (
    GatedConv, GatedDeconv,
    VanillaConv, VanillaDeconv
)

# I added
class SelfAttention(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.key = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.query = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv3d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, D, H, W = x.size()
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        key = key.view(N, C, -1).permute(0, 2, 1)
        query = query.view(N, C, -1)
        value = value.view(N, C, -1)
        value = value.permute(0, 2, 1)
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        attention = torch.bmm(value, attention.permute(0, 2, 1))
        attention = attention.view(N, C, D, H, W)
        return attention
        
###########################
# Encoder/Decoder Modules #
###########################

class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv

class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type, nc_ref):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(
            nc_in + nc_ref, nf * 1, kernel_size=(3, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.self_attention = SelfAttention(nf*4)


        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 4, 4))
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 8, 8))
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 16, 16))
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        attention = self.self_attention(c6)
        c6 = c6 + attention

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        a3 = self.dilated_conv3(a2)
        a4 = self.dilated_conv4(a3)

        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c4, c2  # For skip connection


class AttentionDownSampleModule(DownSampleModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(nc_in, nf, use_bias, norm, conv_by, conv_type)


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type,
                 use_skip_connection=False):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in * 2 if use_skip_connection else nc_in,
            nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        #self.conv_DF = DeformConv3d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        
        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            nf * 4 if use_skip_connection else nf * 2,
            nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(
            nf * 1, nf // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        #self.conv_DF2 = DeformConv3d(nf // 2, nf // 2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.conv11 = self.ConvBlock(
            nf // 2, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)
        self.use_skip_connection = use_skip_connection

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c4, c2 = inp
        if self.use_skip_connection:
            d1 = self.deconv1(self.concat_feature(c8, c4))
            c9 = self.conv9(d1)
            #c_n = self.conv_DF(c9)
            d2 = self.deconv2(self.concat_feature(c9, c2))
        else:
            d1 = self.deconv1(c8)
            c9 = self.conv9(d1)
            #c_n = self.conv_DF(c9)
            d2 = self.deconv2(c9)

        c10 = self.conv10(d2)
        #c_n2 = self.conv_DF2(c10)
        c11 = self.conv11(c10)
        return c11
    
# solve the cuda memory problem and then add the deformable convolution layers 