import torch
import torch.nn as nn

from .efficientnet_utils import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn=nn.ReLU, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = act_fn()

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, act_fn=nn.ReLU, **kwargs):
        super(ConvTranspose2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size,
                stride=stride, bias=bias, **kwargs
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        )

    def forward(self, x):
        return self.block(x)


class UpsampleConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=False, act_fn=nn.ReLU, **kwargs):
        super(UpsampleConv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            act_fn()
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.block(x)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """
    def __init__(self, in_channels, out_channels=None, act_fn=nn.ReLU, **kwargs):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activate = act_fn()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        x = self.bn(x)
        x = self.activate(x)

        return x


def focus(x):
    # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
    patch_top_left = x[..., ::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_bot_right = x[..., 1::2, 1::2]
    x = torch.cat(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        dim=1,
    )
    return x


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x = focus(x)
        return x


def build_block(name, in_channels, out_channels, act_fn=nn.ReLU, norm_layer=nn.BatchNorm2d, **kwargs):
    if name == 'BasicBlock2D':
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    elif name == 'SeparableConvBlock':
        block = [
            Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False),
            Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1),
            norm_layer(num_features=out_channels, momentum=0.01, eps=1e-3),
            act_fn()
        ]
    elif name == 'DeConv2dBlock':
        block = [
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    else:
        raise NotImplementedError

    return block


def build_deconv_block(name, in_channels, out_channels, act_fn, norm_layer=nn.BatchNorm2d, **kwargs):
    if name == 'ConvTranspose2dBlock':
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, **kwargs
            ),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        )
    elif name == 'UpsampleConv2dBlock':
        block = nn.Sequential(
            nn.Upsample(scale_factor=kwargs['stride'], mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=kwargs['bias']),
            norm_layer(out_channels),
            act_fn()
        )
    elif name == 'Conv2dUpsampleBlock':
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=kwargs['bias']),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=kwargs['stride'], mode='bilinear', align_corners=True)
        )
    elif name == 'UpsampleBlock':
        block = nn.Sequential(
            nn.Upsample(scale_factor=kwargs['stride'], mode='bilinear', align_corners=True)
        )
    else:
        raise NotImplementedError

    return block


def build_downsample_block(mode, in_channel, out_channel, norm_layer=nn.BatchNorm2d, **kwargs):
    downsample_block = []
    if mode == 'maxpooling':
        downsample_block.extend([
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        ])
    elif mode == 'avgpooling':
        downsample_block.extend([
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        ])
    elif mode == 'conv':
        downsample_block.extend([
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ])
    elif mode == 'focus':
        downsample_block.extend([
            Focus(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ])
    return nn.Sequential(*downsample_block)
