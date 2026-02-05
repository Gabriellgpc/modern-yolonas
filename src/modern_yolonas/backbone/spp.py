"""Spatial Pyramid Pooling for YoloNAS."""

from __future__ import annotations

from typing import Type

import torch
from torch import nn, Tensor

from modern_yolonas.nn.blocks import Conv


class SPP(nn.Module):
    """SPP as used in YoloNAS backbone context_module.

    Attributes: ``cv1``, ``cv2``, ``m`` (ModuleList of MaxPool2d).
    """

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        k: tuple[int, ...] = (5, 9, 13),
        activation_type: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * (len(k) + 1), output_channels, 1, 1, activation_type)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    @property
    def out_channels(self):
        return self.cv2.conv.out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
