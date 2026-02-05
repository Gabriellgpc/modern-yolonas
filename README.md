# modern-yolonas

A clean, minimal Python reimplementation of [YOLO-NAS](https://github.com/Deci-AI/super-gradients) object detection. No factory patterns, no registries, no OmegaConf — just PyTorch.

## Install

```bash
uv add modern-yolonas
# or
pip install modern-yolonas
```

## Quick Start

### Python API

```python
from modern_yolonas import yolo_nas_s

# Load pretrained COCO model
model = yolo_nas_s(pretrained=True)

# High-level detection
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")
result = det("image.jpg")
result.save("output.jpg")
```

### CLI

```bash
# Detection
yolonas detect --model yolo_nas_s --source image.jpg --conf 0.25

# Training
yolonas train --model yolo_nas_s --data /path/to/dataset --format yolo --epochs 100

# Evaluation
yolonas eval --model yolo_nas_s --data /path/to/coco --split val2017

# Export
yolonas export --model yolo_nas_s --format onnx --output model.onnx
```

## Variants

| Model | Params | Input | mAP (COCO val) |
|---|---|---|---|
| YOLO-NAS S | ~12M | 640 | 47.5 |
| YOLO-NAS M | ~31M | 640 | 51.5 |
| YOLO-NAS L | ~44M | 640 | 52.2 |

## Development

```bash
uv sync --dev
uv run pytest tests/ -v
uv run ruff check src/
```

## License

MIT
