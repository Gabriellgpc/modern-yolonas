# modern-yolonas

A clean, minimal Python reimplementation of [YOLO-NAS](https://github.com/Deci-AI/super-gradients) object detection. No factory patterns, no registries, no OmegaConf — just PyTorch.

## Install

```bash
uv add modern-yolonas
# or
pip install modern-yolonas
```

## Quick Start

### Detect objects in an image

```python
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")
result = det("image.jpg")

# Print detections
from modern_yolonas.inference.visualize import COCO_NAMES

for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
    name = COCO_NAMES[int(cls_id)]
    x1, y1, x2, y2 = box
    print(f"{name}: {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Save annotated image
result.save("output.jpg")
```

### Detect objects in a video

```python
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

# Option 1: Write annotated video directly
stats = det.detect_video_to_file("input.mp4", "output.mp4")
print(f"{stats['total_detections']} detections across {stats['total_frames']} frames")

# Option 2: Iterate frames for custom logic
for frame_idx, result in det.detect_video("input.mp4"):
    print(f"Frame {frame_idx}: {len(result.boxes)} objects")
    # result.boxes, result.scores, result.class_ids are numpy arrays
    # result.visualize() returns the annotated frame as BGR numpy array
```

### Live webcam detection

```python
import cv2
from modern_yolonas import Detector

det = Detector("yolo_nas_s", device="cuda")

for frame_idx, result in det.detect_video(source=0):  # 0 = default camera
    cv2.imshow("YOLO-NAS", result.visualize())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

### Low-level model API

```python
import torch
from modern_yolonas import yolo_nas_s

model = yolo_nas_s(pretrained=True).eval().cuda()
x = torch.randn(1, 3, 640, 640).cuda()
pred_bboxes, pred_scores = model(x)
# pred_bboxes: [1, 8400, 4] — x1y1x2y2 pixel coordinates
# pred_scores: [1, 8400, 80] — class probabilities
```

## CLI

```bash
# Detect in images
yolonas detect --model yolo_nas_s --source image.jpg --conf 0.25
yolonas detect --model yolo_nas_l --source images/ --output results/

# Detect in video
yolonas detect --model yolo_nas_s --source video.mp4 --output results/
yolonas detect --model yolo_nas_m --source video.mp4 --skip-frames 2 --conf 0.3

# Training
yolonas train --model yolo_nas_s --data /path/to/dataset --format yolo --epochs 100

# Evaluation
yolonas eval --model yolo_nas_s --data /path/to/coco --split val2017

# Export
yolonas export --model yolo_nas_s --format onnx --output model.onnx
yolonas export --model yolo_nas_s --format openvino --output model.xml

# Export for Frigate (embeds preprocessing + NMS in the graph)
yolonas export --model yolo_nas_s --format onnx --target frigate
yolonas export --model yolo_nas_s --format openvino --target frigate --input-size 320
```

### Frigate Integration

The `--target frigate` export produces a self-contained model that accepts raw `uint8` BGR
input and outputs a flat `[D, 7]` tensor with `[batch, x1, y1, x2, y2, confidence, class_id]`.

Example Frigate configuration:

```yaml
detectors:
  ov:
    type: openvino
    device: GPU

model:
  model_type: yolonas
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: bgr
  path: /config/model_frigate.xml
```

## Examples

See the [`examples/`](examples/) directory:

- [`detect_image.py`](examples/detect_image.py) — run detection on a single image
- [`detect_video.py`](examples/detect_video.py) — run detection on a video file
- [`detect_webcam.py`](examples/detect_webcam.py) — live webcam detection

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

## Acknowledgments

This project is a clean-room reimplementation of the YOLO-NAS architecture originally developed by [Deci AI](https://deci.ai/) and published in their [super-gradients](https://github.com/Deci-AI/super-gradients) library (Apache-2.0). The model architecture, module structure, and state_dict key naming were derived from the super-gradients source code to enable pretrained weight compatibility.

**Pretrained weights notice:** The pretrained COCO weights downloaded by this library (via `pretrained=True`) are provided by Deci AI and are subject to [Deci's YOLO-NAS license](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md), which restricts commercial use and redistribution. The MIT license of this repository applies only to the source code, **not** to the pretrained weights. If you train your own weights from scratch, those are entirely yours.

## License

MIT — applies to the source code only. See [Acknowledgments](#acknowledgments) for pretrained weight licensing.
