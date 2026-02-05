"""CLI: yolonas detect"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--source", required=True, help="Image file, directory, or video path.")
@click.option("--conf", default=0.25, help="Confidence threshold.")
@click.option("--iou", default=0.45, help="NMS IoU threshold.")
@click.option("--device", default="cuda", help="Device (cuda or cpu).")
@click.option("--output", default="results", help="Output directory.")
@click.option("--input-size", default=640, help="Model input size.")
def detect(model: str, source: str, conf: float, iou: float, device: str, output: str, input_size: int):
    """Run object detection on images or video."""
    from rich.console import Console

    from modern_yolonas.inference.detect import Detector

    console = Console()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Loading {model}...")
    det = Detector(model, device=device, conf_threshold=conf, iou_threshold=iou, input_size=input_size)

    source_path = Path(source)

    if source_path.is_dir():
        files = sorted(source_path.glob("*.*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    else:
        files = [source_path]

    for f in files:
        console.print(f"Processing {f.name}...")
        result = det(str(f))
        out_path = out_dir / f.name
        result.save(out_path)
        console.print(f"  {len(result.boxes)} detections → {out_path}")

    console.print(f"[green]Done! Results saved to {out_dir}[/green]")
