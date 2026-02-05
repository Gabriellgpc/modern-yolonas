"""Optimizer factory."""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW, SGD, Optimizer


def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
) -> Optimizer:
    """Create optimizer with separate weight decay groups.

    BatchNorm and bias parameters do not receive weight decay.
    """
    decay_params = []
    no_decay_params = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, nn.BatchNorm2d) or param_name == "bias":
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name.lower() == "adamw":
        return AdamW(param_groups, lr=lr)
    elif name.lower() == "sgd":
        return SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
