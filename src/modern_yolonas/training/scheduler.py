"""Cosine LR scheduler with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    warmup_lr: float = 1e-6,
    total_steps: int = 100000,
    cosine_final_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create cosine annealing scheduler with linear warmup.

    Args:
        optimizer: Optimizer to schedule.
        warmup_steps: Number of warmup steps.
        warmup_lr: Starting LR during warmup (as fraction of base LR).
        total_steps: Total training steps.
        cosine_final_lr_ratio: Final LR as fraction of initial LR.
    """
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return warmup_lr / base_lrs[0] + (1.0 - warmup_lr / base_lrs[0]) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cosine_final_lr_ratio + 0.5 * (1.0 - cosine_final_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
