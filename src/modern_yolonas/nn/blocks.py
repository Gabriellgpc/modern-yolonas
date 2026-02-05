"""Core convolution building blocks — Conv, ConvBNReLU, autopad.

Attribute names match super-gradients for state_dict key compatibility.
"""

from __future__ import annotations

import math
from typing import Type

from torch import nn


def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


def width_multiplier(original: int, factor: float, divisor: int | None = None) -> int:
    if divisor is None:
        return int(original * factor)
    return math.ceil(int(original * factor) / divisor) * divisor


class Conv(nn.Module):
    """Conv2d + BatchNorm2d + Activation.

    Attributes ``conv``, ``bn``, ``act`` match super-gradients ``Conv``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        activation_type: Type[nn.Module] = nn.ReLU,
        padding: int | None = None,
        groups: int | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel,
            stride,
            autopad(kernel, padding),
            groups=groups or 1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU wrapped in a Sequential named ``seq``.

    Children are named ``seq.conv``, ``seq.bn``, ``seq.act`` to match
    super-gradients state_dict keys.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        use_normalization: bool = True,
        activation_type: Type[nn.Module] | None = nn.ReLU,
        activation_kwargs: dict | None = None,
        inplace: bool = False,
    ):
        super().__init__()
        if activation_kwargs is None:
            activation_kwargs = {}

        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
        )
        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels))
        if activation_type is not None:
            self.seq.add_module("act", activation_type(**activation_kwargs))

    def forward(self, x):
        return self.seq(x)
