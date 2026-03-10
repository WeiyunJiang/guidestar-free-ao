'''A number of custom pytorch modules with sane defaults that I find useful for model prototyping.'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils

import numpy as np

import math
import numbers

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)


# From https://gist.github.com/wassname/ecd2dac6fc8f9918149853d17e3abf02
class LayerNormConv2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class DownBlock3D(nn.Module):
    '''A 3D convolutional downsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=4,
                      padding=0,
                      stride=2,
                      bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class UpBlock3D(nn.Module):
    '''A 3D convolutional upsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.ReLU(True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class Conv3dSame(torch.nn.Module):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReplicationPad3d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb, ka, kb)),
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

    def forward(self, x):
        return self.net(x)


class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
                           bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.ReLU(True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.ReLU(True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels,
                              middle_channels,
                              kernel_size=3,
                              padding=0,
                              stride=1,
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(middle_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        net += [nn.ReflectionPad2d(1),
                nn.Conv2d(middle_channels,
                          out_channels,
                          kernel_size=4,
                          padding=0,
                          stride=2,
                          bias=True if norm is None else False)]

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Unet3d(nn.Module):
    '''A 3d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 norm=nn.BatchNorm3d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet3d."

        # Define the in block
        self.in_layer = [Conv3dSame(in_channels, nf0, kernel_size=3, bias=False)]

        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]

        self.in_layer += [nn.LeakyReLU(0.2, True)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block. The feature map has height and width 1 --> no batchnorm.
        self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    norm=None)
        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** i * nf0, max_channels)),
                                                        int(min(2 ** (i + 1) * nf0, max_channels)),
                                                        submodule=self.unet_block,
                                                        norm=norm)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv3dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear)]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class UnetSkipConnectionBlock3d(nn.Module):
    '''Helper class for building a 3D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 norm=nn.BatchNorm3d,
                 submodule=None):
        super().__init__()

        if submodule is None:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     UpBlock3D(inner_nc, outer_nc, norm=norm)]
        else:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     submodule,
                     UpBlock3D(2 * inner_nc, outer_nc, norm=norm)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class UnetSkipConnectionBlock(nn.Module):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class Unet(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 use_dropout,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv2dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class Identity(nn.Module):
    '''Helper module to allow Downsampling and Upsampling nets to default to identity if they receive an empty list.'''

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class DownsamplingNet(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.downs = list()
            self.downs.append(DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm))
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                self.downs.append(DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm))
            self.downs = nn.Sequential(*self.downs)

    def forward(self, input):
        return self.downs(input)


class UpsamplingNet(nn.Module):
    '''A subnetwork that upsamples a 2D feature map with a variety of upsampling options.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 upsampling_mode,
                 use_dropout,
                 dropout_prob=0.1,
                 first_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of upsampling steps (each step upsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param upsampling_mode: Mode of upsampling. For documentation, see class "UpBlock"
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param first_layer_one: Whether the input to the last layer will have a spatial size of 1. In that case,
                               the first layer will not have a norm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.ups = Identity()
        else:
            self.ups = list()
            self.ups.append(UpBlock(in_channels,
                                    per_layer_out_ch[0],
                                    use_dropout=use_dropout,
                                    dropout_prob=dropout_prob,
                                    norm=None if first_layer_one else norm,
                                    upsampling_mode=upsampling_mode))
            for i in range(0, len(per_layer_out_ch) - 1):
                self.ups.append(
                    UpBlock(per_layer_out_ch[i],
                            per_layer_out_ch[i + 1],
                            use_dropout=use_dropout,
                            dropout_prob=dropout_prob,
                            norm=norm,
                            upsampling_mode=upsampling_mode))
            self.ups = nn.Sequential(*self.ups)

    def forward(self, input):
        return self.ups(input)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.leaky_relu(self.bn(self.conv1(x)))
        x = self.leaky_relu(self.bn(self.conv2(x)))
        return x

class BlurKernelUNet64x64(nn.Module):
    def __init__(self):
        super(BlurKernelUNet64x64, self).__init__()

        # Downsampling path
        self.enc1 = UNetBlock(1, 32)   # Input: 256x256, change 64 to 32
        self.enc2 = UNetBlock(32, 64)  # Input: 128x128, change input to 32
        self.enc3 = UNetBlock(64, 128)  # Input: 64x64, no change
        self.enc4 = UNetBlock(128, 256)  # Input: 32x32, no change
        
        # Strided convolutions for downsampling
        self.downsample1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # Output: 128x128, change input/output to 32
        self.downsample2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Output: 64x64
        self.downsample3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # Output: 32x32
        self.downsample4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # Output: 16x16
        
        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)  # Input: 16x16, change input/output to 256/512
        
        # Upsampling path
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 32x32
        self.dec4 = UNetBlock(512, 256)  # Change input to 512 (256 + 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 64x64
        self.dec3 = UNetBlock(256, 128)  # Change input to 256 (128 + 128)

        # Correct upsampling to ensure the final output is 64x64
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 128x128
        self.dec2 = UNetBlock(128, 64)  # Adjusted input

        # Final upsample and ensure output is 64x64
        self.final_down = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)  # Output: 32x32 # Output: 64x64
        self.conv_final = nn.Conv2d(32, 1, kernel_size=1)  # 64x64x1 output
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        # Downsampling path using strided convolutions
        enc1 = self.enc1(x)  # 256x256
        # print(f'enc1: {enc1.shape}')
        enc2 = self.enc2(self.leaky_relu(self.downsample1(enc1)))  # 128x128
        # print(f'enc2: {enc2.shape}')
        enc3 = self.enc3(self.leaky_relu(self.downsample2(enc2)))  # 64x64
        # print(f'enc3: {enc3.shape}')
        enc4 = self.enc4(self.leaky_relu(self.downsample3(enc3)))  # 32x32
        # print(f'enc4: {enc4.shape}')
        
        # Bottleneck
        bottleneck = self.bottleneck(self.leaky_relu(self.downsample4(enc4)))  # 16x16
        # print(f'bottleneck: {bottleneck.shape}')
        
        # Upsampling path
        dec4 = self.upconv4(bottleneck)  # 32x32
        # print(f'dec4: {dec4.shape}')
        dec4 = torch.cat((enc4, dec4), dim=1)
        # print(f'dec4: {dec4.shape}')
        dec4 = self.dec4(dec4)
        # print(f'dec4: {dec4.shape}')
        
        dec3 = self.upconv3(dec4)  # 64x64
        # print(f'dec3: {dec3.shape}')
        dec3 = torch.cat((enc3, dec3), dim=1)
        # print(f'dec3: {dec3.shape}')
        dec3 = self.dec3(dec3)
        # print(f'dec3: {dec3.shape}')
        
        # Upsample to 128x128
        dec2 = self.upconv2(dec3)  # 128x128
        # print(f'dec2: {dec2.shape}')
        dec2 = torch.cat((enc2, dec2), dim=1)
        # print(f'dec2: {dec2.shape}')
        dec2 = self.dec2(dec2)
        # print(f'dec2: {dec2.shape}')

        # Ensure final output is 64x64
        final_down = self.final_down(dec2)  # 64x64
        # print(f'final_down: {final_down.shape}')
        output = self.conv_final(final_down)  # 64x64x1
        
        return output

if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # model = Unet(in_channels=1,out_channels=1,nf0=32,num_down=5,max_channels=512,use_dropout=False,outermost_linear=True).to(device)
    # psf = torch.randn(32,1,256,256).to(device)
    # out = model(psf)
    # print(out.shape)    

    # Example usage:
    model = BlurKernelUNet64x64().to(device)
    input_tensor = torch.randn(32, 1, 256, 256).to(device)  # Example input: batch size 1, grayscale image 256x256
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Should be (1, 1, 64, 64)
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")