"""YOLO txt-label format dataset.

Expected layout::

    root/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   └── val/
    └── labels/
        ├── train/
        │   ├── img001.txt
        │   └── ...
        └── val/

Each label file has one row per object: ``class_id x_center y_center width height`` (normalized).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class YOLODetectionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms=None,
        input_size: int = 640,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.input_size = input_size

        img_dir = self.root / "images" / split
        label_dir = self.root / "labels" / split

        self.images = sorted(img_dir.glob("*.*"))
        self.images = [p for p in self.images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        self.label_dir = label_dir

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, img_path: Path) -> Path:
        return self.label_dir / (img_path.stem + ".txt")

    def load_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and targets without transforms."""
        img_path = self.images[index]
        image = cv2.imread(str(img_path))

        label_path = self._label_path(img_path)
        if label_path.exists():
            targets = np.loadtxt(str(label_path), ndmin=2).reshape(-1, 5)
        else:
            targets = np.zeros((0, 5))

        return image, targets.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, targets = self.load_raw(index)
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        return image, targets
