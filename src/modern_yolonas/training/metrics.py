"""COCO mAP evaluation using pycocotools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from torch import Tensor


class COCOEvaluator:
    """Wrapper around pycocotools for mAP computation.

    Args:
        ann_file: Path to COCO annotations JSON.
    """

    def __init__(self, ann_file: str | Path):
        from pycocotools.coco import COCO

        self.coco_gt = COCO(str(ann_file))
        self.results: list[dict] = []

        # Build reverse mapping: label → category_id
        cat_ids = sorted(self.coco_gt.getCatIds())
        self.label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    def reset(self):
        self.results = []

    def update(
        self,
        image_ids: list[int],
        boxes: list[Tensor],
        scores: list[Tensor],
        class_ids: list[Tensor],
    ):
        """Add batch of predictions.

        Args:
            image_ids: COCO image IDs.
            boxes: List of ``[D, 4]`` tensors (x1y1x2y2 pixel coords).
            scores: List of ``[D]`` confidence tensors.
            class_ids: List of ``[D]`` integer class ID tensors.
        """
        for img_id, bxs, scs, cids in zip(image_ids, boxes, scores, class_ids):
            for box, score, cls_id in zip(bxs, scs, cids):
                x1, y1, x2, y2 = box.tolist()
                self.results.append({
                    "image_id": int(img_id),
                    "category_id": self.label_to_cat_id.get(int(cls_id), int(cls_id)),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: x, y, w, h
                    "score": float(score),
                })

    def evaluate(self) -> dict[str, float]:
        """Compute COCO metrics.

        Returns:
            Dict with ``mAP``, ``mAP_50``, ``mAP_75``, etc.
        """
        from pycocotools.cocoeval import COCOeval

        if not self.results:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results, f)
            tmp_path = f.name

        coco_dt = self.coco_gt.loadRes(tmp_path)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        Path(tmp_path).unlink(missing_ok=True)

        return {
            "mAP": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
        }
