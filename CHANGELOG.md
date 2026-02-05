# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- OpenVINO export support (`yolonas export --format openvino`)
- Automated PyPI publishing via GitHub Actions (trusted publishing)
- Dynamic versioning via `hatch-vcs` (git tag based)

## [0.1.0] - 2025-06-01

### Added
- YOLO-NAS S/M/L model architectures (QARepVGG backbone, PAN neck, DFL heads)
- Pretrained weight loading from super-gradients checkpoints (`strict=True` compatible)
- Inference pipeline: preprocessing, postprocessing with NMS, visualization
- Image detection with bounding box overlay and class labels
- Video detection with frame iteration and direct file output
- COCO and YOLO format dataset loaders with mosaic/mixup augmentations
- Training with DDP, AMP, EMA, cosine LR scheduler, PPYoloE loss
- COCO mAP evaluation
- ONNX export with dynamic batch axes
- Click-based CLI: `detect`, `train`, `eval`, `export`
- CI with GitHub Actions (lint + test)
