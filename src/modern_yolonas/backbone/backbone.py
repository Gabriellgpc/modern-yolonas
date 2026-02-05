"""YoloNAS backbone — stem + stages + SPP context module.

Replaces the factory-driven NStageBackbone from super-gradients.
Attribute names ``stem``, ``stage1`` … ``stage4``, ``context_module``
mirror super-gradients for state_dict key compatibility.
"""

from __future__ import annotations

from typing import Type

from torch import nn, Tensor

from modern_yolonas.backbone.stem import YoloNASStem
from modern_yolonas.backbone.stage import YoloNASStage
from modern_yolonas.backbone.spp import SPP


class YoloNASBackbone(nn.Module):
    """Assembles stem → stage1 → stage2 → stage3 → stage4 + SPP.

    Returns ``[c2, c3, c4, c5]`` corresponding to ``[stage1, stage2, stage3, context_module]``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_out_channels: int = 48,
        stages_config: list[dict] | None = None,
        spp_output_channels: int = 768,
        spp_k: tuple[int, ...] = (5, 9, 13),
        activation_type: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        if stages_config is None:
            raise ValueError("stages_config is required")

        self.stem = YoloNASStem(in_channels, stem_out_channels)

        prev_channels = stem_out_channels
        for i, cfg in enumerate(stages_config, start=1):
            stage = YoloNASStage(
                in_channels=prev_channels,
                out_channels=cfg["out_channels"],
                num_blocks=cfg["num_blocks"],
                activation_type=activation_type,
                hidden_channels=cfg.get("hidden_channels"),
                concat_intermediates=cfg.get("concat_intermediates", False),
            )
            setattr(self, f"stage{i}", stage)
            prev_channels = cfg["out_channels"]

        self.context_module = SPP(prev_channels, spp_output_channels, k=spp_k, activation_type=activation_type)

        # out_layers = [stage1, stage2, stage3, context_module]
        self._out_channels = [
            stages_config[0]["out_channels"],  # c2
            stages_config[1]["out_channels"],  # c3
            stages_config[2]["out_channels"],  # c4
            spp_output_channels,  # c5
        ]

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        outputs = []
        for i in range(1, 5):
            x = getattr(self, f"stage{i}")(x)
            if i >= 1:
                outputs.append(x)
        # Replace last output with SPP output
        outputs[-1] = self.context_module(outputs[-1])
        return outputs
