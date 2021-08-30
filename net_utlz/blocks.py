import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional
import torch.nn.functional as F
from net_utlz.utlz import initialize_weights_xavier


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, style_size=6):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        self.transform = Transform(out_size, style_size)
        self.model = nn.ModuleList(layers)

    def forward(self, x, skip_input, prob_t):
        skip_input = self.transform(skip_input, prob_t)

        for operation in self.model:
                x = operation(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Transform(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style_1d = nn.Linear(style_dim, in_channel * 2, bias=True)
        # self.style_2d = nn.Sequential(
        #     nn.Conv2d(in_channel * 2, in_channel * 2, 1, 1, 0, bias=True),
        #     nn.InstanceNorm2d(in_channel * 2),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channel * 2, in_channel * 2, 1, 1, 0, bias=True)
        # )
        self.noise = NoiseInjection()
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.style_1d], 0.1)
        # initialize_weights_xavier(self.style_2d, 0)

    def forward(self, feat, style):
        style = self.style_1d(style).unsqueeze(2).unsqueeze(3)
        # _, _, h, w = feat.size()
        # style = F.interpolate(style, (h, w), mode="bilinear", align_corners=True)
        # style = self.style_2d(style)
        gamma, beta = style.chunk(2, 1)
        noisy = self.noise(feat)

        out = self.norm(noisy)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise
