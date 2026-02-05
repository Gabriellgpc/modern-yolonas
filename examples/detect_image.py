"""Example: Run YOLO-NAS inference on a single image.

Usage:
    python examples/detect_image.py path/to/image.jpg
    python examples/detect_image.py path/to/image.jpg --model yolo_nas_l --device cpu
"""

import argparse

from modern_yolonas.inference.detect import Detector


def main():
    parser = argparse.ArgumentParser(description="YOLO-NAS image detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    args = parser.parse_args()

    # Create detector (downloads pretrained weights on first run)
    det = Detector(args.model, device=args.device, conf_threshold=args.conf, iou_threshold=args.iou)

    # Run detection
    result = det(args.image)

    # Print results
    print(f"Found {len(result.boxes)} objects:")
    from modern_yolonas.inference.visualize import COCO_NAMES

    for box, score, cls_id in zip(result.boxes, result.scores, result.class_ids):
        name = COCO_NAMES[int(cls_id)]
        x1, y1, x2, y2 = box
        print(f"  {name}: {score:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # Save annotated image
    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
