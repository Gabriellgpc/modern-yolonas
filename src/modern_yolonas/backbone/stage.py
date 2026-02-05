"""YoloNAS backbone stage components.

Classes:
  YoloNASBottleneck — two QARepVGGBlocks + optional residual
  SequentialWithIntermediates — returns all intermediate outputs
  YoloNASCSPLayer — cross-stage partial layer
  YoloNASStage — downsample + CSP layer
"""

from __future__ import annotations

from typing import Type, Iterable

import torch
from torch import nn, Tensor

from modern_yolonas.nn.blocks import Conv
from modern_yolonas.nn.drop_path import DropPath
from modern_yolonas.nn.repvgg import QARepVGGBlock, Residual


class YoloNASBottleneck(nn.Module):
    """Two consecutive blocks with optional residual.

    Attributes match super-gradients: ``cv1``, ``cv2``, ``shortcut``, ``drop_path``, ``alpha``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        block_type: Type[nn.Module],
        activation_type: Type[nn.Module],
        shortcut: bool,
        use_alpha: bool,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.cv1 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.cv2 = block_type(output_channels, output_channels, activation_type=activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

    def forward(self, x: Tensor) -> Tensor:
        y = self.drop_path(self.cv2(self.cv1(x)))
        return self.alpha * self.shortcut(x) + y if self.add else y


class SequentialWithIntermediates(nn.Sequential):
    """A Sequential that can optionally return all intermediate values."""

    def __init__(self, output_intermediates: bool, *args):
        super().__init__(*args)
        self.output_intermediates = output_intermediates

    def forward(self, input: Tensor) -> list[Tensor]:
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        return [super().forward(input)]


class YoloNASCSPLayer(nn.Module):
    """Cross-stage partial layer.

    Attributes: ``conv1``, ``conv2``, ``conv3``, ``bottlenecks``, ``dropout``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        block_type: Type[nn.Module],
        activation_type: Type[nn.Module],
        shortcut: bool = True,
        use_alpha: bool = True,
        expansion: float = 0.5,
        hidden_channels: int | None = None,
        concat_intermediates: bool = False,
        drop_path_rates: Iterable[float] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_bottlenecks
        else:
            drop_path_rates = list(drop_path_rates)

        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv3 = Conv(
            hidden_channels * (2 + concat_intermediates * num_bottlenecks),
            out_channels,
            1,
            stride=1,
            activation_type=activation_type,
        )

        module_list = [
            YoloNASBottleneck(
                hidden_channels,
                hidden_channels,
                block_type,
                activation_type,
                shortcut,
                use_alpha,
                drop_path_rate=drop_path_rates[i],
            )
            for i in range(num_bottlenecks)
        ]
        self.bottlenecks = SequentialWithIntermediates(concat_intermediates, *module_list)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        x = self.dropout(x)
        return self.conv3(x)


class YoloNASStage(nn.Module):
    """Single backbone stage: QARepVGGBlock downsample → YoloNASCSPLayer.

    Attributes: ``downsample``, ``blocks``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation_type: Type[nn.Module] = nn.ReLU,
        hidden_channels: int | None = None,
        concat_intermediates: bool = False,
        drop_path_rates: Iterable[float] | None = None,
        dropout_rate: float = 0.0,
        stride: int = 2,
    ):
        super().__init__()
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(
            in_channels, out_channels, stride=stride, activation_type=activation_type, use_residual_connection=False
        )
        self.blocks = YoloNASCSPLayer(
            out_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            True,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.downsample(x))
