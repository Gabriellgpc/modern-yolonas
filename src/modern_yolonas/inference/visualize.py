"""Draw detection boxes on images."""

from __future__ import annotations

import numpy as np

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Deterministic color per class (BGR)."""
    # Simple hash-based palette
    r = ((class_id * 47) % 200) + 55
    g = ((class_id * 97) % 200) + 55
    b = ((class_id * 157) % 200) + 55
    return (int(b), int(g), int(r))


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: HWC uint8 BGR image (modified in-place and returned).
        boxes: ``[D, 4]`` x1y1x2y2.
        scores: ``[D]`` confidence scores.
        class_ids: ``[D]`` integer class IDs.
        class_names: List of class names (defaults to COCO).
        thickness: Box line thickness.
        font_scale: Label font scale.

    Returns:
        Annotated image.
    """
    import cv2

    if class_names is None:
        class_names = COCO_NAMES

    image = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        color = _color_for_class(cls_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return image
