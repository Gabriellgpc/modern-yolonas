"""High-level detection API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from modern_yolonas.inference.preprocess import preprocess
from modern_yolonas.inference.postprocess import postprocess, rescale_boxes
from modern_yolonas.inference.visualize import draw_detections


@dataclass
class Detection:
    """Detection results for a single image."""

    boxes: np.ndarray  # [D, 4] x1y1x2y2
    scores: np.ndarray  # [D]
    class_ids: np.ndarray  # [D]
    image: np.ndarray | None = field(default=None, repr=False)

    def visualize(self, class_names: list[str] | None = None) -> np.ndarray:
        """Draw detections on the original image."""
        if self.image is None:
            raise ValueError("Original image not stored; pass retain_image=True to Detector")
        return draw_detections(self.image, self.boxes, self.scores, self.class_ids, class_names)

    def save(self, path: str | Path, class_names: list[str] | None = None):
        """Visualize and save to file."""
        import cv2

        img = self.visualize(class_names)
        cv2.imwrite(str(path), img)


class Detector:
    """High-level detector: load model → preprocess → forward → postprocess.

    Usage::

        det = Detector("yolo_nas_s", device="cuda")
        results = det("image.jpg")
        results.save("output.jpg")
    """

    def __init__(
        self,
        model: str = "yolo_nas_s",
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        pretrained: bool = True,
    ):
        from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

        builders = {
            "yolo_nas_s": yolo_nas_s,
            "yolo_nas_m": yolo_nas_m,
            "yolo_nas_l": yolo_nas_l,
        }
        if model not in builders:
            raise ValueError(f"Unknown model: {model}. Choose from {list(builders)}")

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self.model = builders[model](pretrained=pretrained).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        source: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        retain_image: bool = True,
    ) -> Detection:
        """Run detection on a single image.

        Args:
            source: File path or BGR numpy array.
            conf_threshold: Override instance default.
            iou_threshold: Override instance default.
            retain_image: Store original image in result for visualization.
        """
        import cv2

        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            image = source

        tensor, scale, pad = preprocess(image, self.input_size)
        tensor = tensor.to(self.device)

        pred_bboxes, pred_scores = self.model(tensor)

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        results = postprocess(pred_bboxes, pred_scores, conf, iou)

        boxes, scores, class_ids = results[0]
        boxes = rescale_boxes(boxes, scale, pad, image.shape[:2])

        return Detection(
            boxes=boxes.cpu().numpy(),
            scores=scores.cpu().numpy(),
            class_ids=class_ids.cpu().numpy(),
            image=image if retain_image else None,
        )
