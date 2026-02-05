"""Exponential Moving Average for model parameters."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Args:
        model: The model to track.
        decay: EMA decay factor.
        warmup_steps: Steps before reaching full decay.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9997, warmup_steps: int = 2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        model_sd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(model_sd[k], alpha=1.0 - d)

    @torch.no_grad()
    def state_dict(self) -> dict:
        return {
            "ema_state_dict": self.ema.state_dict(),
            "decay": self.decay,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict):
        self.ema.load_state_dict(state["ema_state_dict"])
        self.decay = state["decay"]
        self.updates = state["updates"]
