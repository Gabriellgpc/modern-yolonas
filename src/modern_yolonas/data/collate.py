"""Detection batch collation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def detection_collate_fn(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor]:
    """Collate detection samples into a batch.

    Each sample is ``(image, targets)`` where targets is ``[N_i, 5]``
    with columns ``[class_id, x_center, y_center, w, h]`` (normalized).

    Returns:
        images: ``[B, 3, H, W]`` float32 tensor.
        targets: ``[sum(N_i), 6]`` tensor with columns ``[batch_idx, class_id, x, y, w, h]``.
    """
    images = []
    targets_list = []

    for i, (img, targets) in enumerate(batch):
        images.append(torch.from_numpy(img))
        if len(targets):
            t = torch.from_numpy(targets).float()
            batch_idx = torch.full((t.shape[0], 1), i, dtype=torch.float32)
            targets_list.append(torch.cat([batch_idx, t], dim=1))

    images = torch.stack(images, 0)
    if targets_list:
        targets = torch.cat(targets_list, 0)
    else:
        targets = torch.zeros(0, 6)

    return images, targets
