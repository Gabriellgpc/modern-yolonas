"""YoloNAS stem — single QARepVGGBlock with stride=2."""

from torch import nn, Tensor

from modern_yolonas.nn.repvgg import QARepVGGBlock


class YoloNASStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(in_channels, out_channels, stride=stride, use_residual_connection=False)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
