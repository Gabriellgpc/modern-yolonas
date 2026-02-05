"""PPYoloE loss for YOLO-NAS training.

Components:
- TaskAlignedAssigner: dynamic label assignment
- VarifocalLoss: classification loss
- GIoULoss: box regression loss
- DFLLoss: distribution focal loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def bbox_iou(box1: Tensor, box2: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute IoU between two sets of boxes (x1y1x2y2 format).

    Args:
        box1: ``[N, 4]``
        box2: ``[M, 4]``

    Returns:
        ``[N, M]`` IoU matrix.
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + eps)


def batch_distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Convert distance predictions (l, t, r, b) to bounding boxes (x1, y1, x2, y2).

    Args:
        points: ``[N, 2]`` anchor points (x, y).
        distance: ``[B, N, 4]`` distances.

    Returns:
        ``[B, N, 4]`` bounding boxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------


class TaskAlignedAssigner:
    """Dynamic label assignment based on task alignment metric.

    Computes alignment metric = score^alpha * iou^beta and selects
    top-K anchors per GT as positive samples.
    """

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def assign(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground truths to anchors.

        Args:
            pred_scores: ``[B, N, C]`` predicted class scores (sigmoid).
            pred_bboxes: ``[B, N, 4]`` predicted boxes (x1y1x2y2).
            anchor_points: ``[N, 2]``.
            gt_labels: ``[B, max_gt, 1]`` class labels.
            gt_bboxes: ``[B, max_gt, 4]`` ground truth boxes (x1y1x2y2).
            mask_gt: ``[B, max_gt, 1]`` valid GT mask.

        Returns:
            assigned_labels, assigned_bboxes, assigned_scores, fg_mask
        """
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        num_max_boxes = gt_bboxes.shape[1]

        if num_max_boxes == 0:
            device = pred_scores.device
            return (
                torch.full([batch_size, num_anchors], 0, dtype=torch.long, device=device),
                torch.zeros([batch_size, num_anchors, 4], device=device),
                torch.zeros([batch_size, num_anchors, pred_scores.shape[-1]], device=device),
                torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=device),
            )

        # Check which anchors are inside GT boxes
        # anchor_points: [N, 2], gt_bboxes: [B, M, 4]
        lt = anchor_points[None, :, None, :] - gt_bboxes[:, None, :, :2]  # [B, N, M, 2]
        rb = gt_bboxes[:, None, :, 2:] - anchor_points[None, :, None, :]  # [B, N, M, 2]
        bbox_deltas = torch.cat([lt, rb], dim=-1)  # [B, N, M, 4]
        mask_in_gts = bbox_deltas.amin(dim=-1) > self.eps  # [B, N, M]

        # Compute alignment metric
        # Get predicted score for GT class
        gt_labels_expanded = gt_labels.squeeze(-1).long()  # [B, M]

        # Clamp to valid range
        gt_labels_clamped = gt_labels_expanded[:, None, :].expand(-1, num_anchors, -1)
        gt_labels_clamped = gt_labels_clamped.clamp(0, pred_scores.shape[-1] - 1)

        # Gather predicted scores for GT classes: [B, N, M]
        pred_scores_for_gt = pred_scores.gather(
            2,
            gt_labels_clamped.reshape(batch_size, num_anchors, num_max_boxes)
            if gt_labels_clamped.ndim == 3
            else gt_labels_clamped,
        )

        # IoU between predictions and GTs: [B, N, M]
        pair_wise_ious = torch.zeros([batch_size, num_anchors, num_max_boxes], device=pred_bboxes.device)
        for b in range(batch_size):
            pair_wise_ious[b] = bbox_iou(pred_bboxes[b], gt_bboxes[b])

        # Alignment metric
        alignment_metric = pred_scores_for_gt.pow(self.alpha) * pair_wise_ious.pow(self.beta)
        alignment_metric *= mask_in_gts.float()
        alignment_metric *= mask_gt.permute(0, 2, 1).float()

        # Select top-K
        topk_mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
        topk_metrics = torch.zeros_like(alignment_metric)
        for b in range(batch_size):
            for m in range(num_max_boxes):
                if mask_gt[b, m, 0] == 0:
                    continue
                vals = alignment_metric[b, :, m]
                k = min(self.topk, (vals > 0).sum().item())
                if k > 0:
                    topk_vals, topk_idx = vals.topk(k)
                    topk_mask[b, topk_idx, m] = True
                    topk_metrics[b, topk_idx, m] = topk_vals

        # Resolve conflicts: if anchor assigned to multiple GTs, pick highest metric
        mask_pos = topk_mask  # [B, N, M]
        fg_mask = mask_pos.any(dim=-1)  # [B, N]

        # For each positive anchor, pick the GT with highest metric
        max_metric, max_gt_idx = topk_metrics.max(dim=-1)  # [B, N]

        # Assigned labels and bboxes
        assigned_labels = torch.zeros([batch_size, num_anchors], dtype=torch.long, device=pred_scores.device)
        assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=pred_scores.device)
        assigned_scores = torch.zeros_like(pred_scores)

        for b in range(batch_size):
            fg = fg_mask[b]
            if fg.any():
                gt_idx = max_gt_idx[b, fg]
                assigned_labels[b, fg] = gt_labels_expanded[b, gt_idx]
                assigned_bboxes[b, fg] = gt_bboxes[b, gt_idx]

                # Normalize assigned scores by max alignment metric per GT
                for m in range(num_max_boxes):
                    pos_m = mask_pos[b, :, m]
                    if pos_m.any():
                        max_iou = pair_wise_ious[b, pos_m, m].max()
                        max_metric_m = topk_metrics[b, pos_m, m].max()
                        if max_metric_m > 0:
                            norm_metric = topk_metrics[b, :, m] / (max_metric_m + self.eps) * max_iou
                            assigned_scores[b, :, gt_labels_expanded[b, m]] = torch.where(
                                mask_pos[b, :, m],
                                torch.max(assigned_scores[b, :, gt_labels_expanded[b, m]], norm_metric),
                                assigned_scores[b, :, gt_labels_expanded[b, m]],
                            )

        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


class VarifocalLoss(nn.Module):
    """Varifocal loss from VarifocalNet."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score: Tensor, gt_score: Tensor, label: Tensor) -> Tensor:
        """
        Args:
            pred_score: ``[B, N, C]`` predicted logits (pre-sigmoid).
            gt_score: ``[B, N, C]`` soft target scores.
            label: ``[B, N, C]`` binary labels (1 for positive).
        """
        pred_sigmoid = pred_score.sigmoid()
        weight = self.alpha * pred_sigmoid.pow(self.gamma) * (1 - label) + gt_score * label
        bce = F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction="none")
        return (weight * bce).sum()


class GIoULoss(nn.Module):
    """Generalized IoU loss."""

    def forward(self, pred_bboxes: Tensor, target_bboxes: Tensor) -> Tensor:
        """
        Args:
            pred_bboxes: ``[N, 4]`` x1y1x2y2.
            target_bboxes: ``[N, 4]`` x1y1x2y2.
        """
        # Intersection
        inter_x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        inter_y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        inter_x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        inter_y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        area1 = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        area2 = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        union = area1 + area2 - inter

        iou = inter / (union + 1e-9)

        # Enclosing box
        enc_x1 = torch.min(pred_bboxes[:, 0], target_bboxes[:, 0])
        enc_y1 = torch.min(pred_bboxes[:, 1], target_bboxes[:, 1])
        enc_x2 = torch.max(pred_bboxes[:, 2], target_bboxes[:, 2])
        enc_y2 = torch.max(pred_bboxes[:, 3], target_bboxes[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        giou = iou - (enc_area - union) / (enc_area + 1e-9)
        return (1.0 - giou).mean()


class DFLLoss(nn.Module):
    """Distribution Focal Loss for fine-grained box regression."""

    def forward(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred_dist: ``[N, 4*(reg_max+1)]`` raw distribution predictions.
            target: ``[N, 4]`` target distances (float, in [0, reg_max] range).
        """
        reg_max = pred_dist.shape[-1] // 4 - 1
        pred_dist = pred_dist.reshape(-1, reg_max + 1)
        target = target.reshape(-1)

        target_left = target.long().clamp(0, reg_max)
        target_right = (target_left + 1).clamp(0, reg_max)
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right

        loss = (
            F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
            + F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# Combined PPYoloE loss
# ---------------------------------------------------------------------------


class PPYoloELoss(nn.Module):
    """Combined loss for YOLO-NAS training.

    Components:
    - VarifocalLoss (classification)
    - GIoULoss (box regression)
    - DFLLoss (distribution focal loss)

    Weighted sum: cls_weight * vfl + iou_weight * giou + dfl_weight * dfl

    Args:
        num_classes: Number of object classes.
        reg_max: Distribution regression maximum.
        cls_weight: Classification loss weight.
        iou_weight: Box regression loss weight.
        dfl_weight: Distribution focal loss weight.
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        cls_weight: float = 1.0,
        iou_weight: float = 2.5,
        dfl_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.dfl_weight = dfl_weight

        self.assigner = TaskAlignedAssigner()
        self.vfl = VarifocalLoss()
        self.giou_loss = GIoULoss()
        self.dfl_loss = DFLLoss()

    def _bbox2dist(self, anchor_points: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Convert bounding boxes to distances from anchor points."""
        x1y1 = anchor_points - gt_bboxes[..., :2]
        x2y2 = gt_bboxes[..., 2:] - anchor_points
        dist = torch.cat([x1y1, x2y2], dim=-1)
        return dist.clamp(0, self.reg_max - 0.01)

    def forward(
        self,
        predictions: tuple,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss.

        Args:
            predictions: ``(decoded_predictions, raw_predictions)`` from NDFLHeads in training mode.
                decoded_predictions: ``(pred_bboxes [B,N,4], pred_scores [B,N,C])``
                raw_predictions: ``(cls_logits [B,N,C], reg_distri [B,N,4*(reg_max+1)],
                                   anchors, anchor_points, num_anchors_list, stride_tensor)``
            targets: ``[sum(N_i), 6]`` with ``[batch_idx, class_id, x, y, w, h]`` (normalized xywh).

        Returns:
            (total_loss, loss_dict)
        """
        (pred_bboxes_decoded, pred_scores_decoded), (
            cls_logits,
            reg_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions

        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        # Prepare GT in format expected by assigner
        gt_labels_list = []
        gt_bboxes_list = []
        for b in range(batch_size):
            mask = targets[:, 0] == b
            if mask.any():
                t = targets[mask]
                gt_labels_list.append(t[:, 1:2])  # [N_b, 1]
                # Convert normalized xywh to pixel x1y1x2y2
                xc, yc, w, h = t[:, 2], t[:, 3], t[:, 4], t[:, 5]
                # These are normalized; scale by stride_tensor range
                # Actually we need image size — but we work in feature-map coords
                # The pred_bboxes_decoded are already in input-image pixel coords
                # So we convert GT to pixel coords assuming 640x640 (standard)
                # This is handled by the dataloader / collation to provide pixel-coords targets
                gt_bboxes_list.append(torch.stack([
                    xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                ], dim=-1))
            else:
                gt_labels_list.append(torch.zeros(0, 1, device=device))
                gt_bboxes_list.append(torch.zeros(0, 4, device=device))

        # Pad to max GT count
        max_gt = max(len(g) for g in gt_labels_list)
        if max_gt == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {"cls_loss": 0.0, "iou_loss": 0.0, "dfl_loss": 0.0, "total_loss": 0.0}

        gt_labels = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)

        for b in range(batch_size):
            n = len(gt_labels_list[b])
            if n > 0:
                gt_labels[b, :n] = gt_labels_list[b]
                gt_bboxes[b, :n] = gt_bboxes_list[b]
                mask_gt[b, :n] = 1.0

        # Run assigner
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = self.assigner.assign(
            pred_scores_decoded,
            pred_bboxes_decoded,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        num_pos = fg_mask.sum().clamp(min=1).item()

        # Classification loss (VFL)
        one_hot = torch.zeros_like(cls_logits)
        for b in range(batch_size):
            fg = fg_mask[b]
            if fg.any():
                one_hot[b, fg] = assigned_scores[b, fg]

        cls_loss = self.vfl(cls_logits, assigned_scores, (assigned_scores > 0).float()) / num_pos

        # Box regression loss (GIoU) — only on positive anchors
        if fg_mask.any():
            pos_pred_bboxes = pred_bboxes_decoded[fg_mask]
            pos_target_bboxes = assigned_bboxes[fg_mask]
            iou_loss = self.giou_loss(pos_pred_bboxes, pos_target_bboxes)
        else:
            iou_loss = torch.tensor(0.0, device=device)

        # DFL loss — only on positive anchors
        if fg_mask.any():
            pos_reg_distri = reg_distri[fg_mask]  # [num_pos, 4*(reg_max+1)]
            pos_anchor_points = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]  # [num_pos, 2]
            pos_stride = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]  # [num_pos, 1]
            pos_target_bboxes = assigned_bboxes[fg_mask] / pos_stride  # Normalize to feature-map scale
            pos_anchor_points_scaled = pos_anchor_points / pos_stride.squeeze(-1)
            target_dist = self._bbox2dist(pos_anchor_points_scaled, pos_target_bboxes)
            dfl_loss = self.dfl_loss(pos_reg_distri, target_dist)
        else:
            dfl_loss = torch.tensor(0.0, device=device)

        total_loss = self.cls_weight * cls_loss + self.iou_weight * iou_loss + self.dfl_weight * dfl_loss

        loss_dict = {
            "cls_loss": cls_loss.item(),
            "iou_loss": iou_loss.item(),
            "dfl_loss": dfl_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
