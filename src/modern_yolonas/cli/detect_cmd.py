"""CLI: yolonas detect"""

from __future__ import annotations

from pathlib import Path

import click

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--source", required=True, help="Image file, directory, or video path.")
@click.option("--conf", default=0.25, help="Confidence threshold.")
@click.option("--iou", default=0.45, help="NMS IoU threshold.")
@click.option("--device", default="cuda", help="Device (cuda or cpu).")
@click.option("--output", default="results", help="Output directory.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--skip-frames", default=0, help="Process every N-th frame for video (0 = every frame).")
@click.option("--codec", default="mp4v", help="Video output codec (e.g. mp4v, XVID, avc1).")
def detect(
    model: str,
    source: str,
    conf: float,
    iou: float,
    device: str,
    output: str,
    input_size: int,
    skip_frames: int,
    codec: str,
):
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
        # Directory of images
        files = sorted(source_path.glob("*.*"))
        files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
        _detect_images(det, files, out_dir, console)

    elif source_path.suffix.lower() in VIDEO_EXTENSIONS:
        # Video file
        _detect_video(det, source_path, out_dir, console, skip_frames, codec)

    elif source_path.suffix.lower() in IMAGE_EXTENSIONS:
        # Single image
        _detect_images(det, [source_path], out_dir, console)

    else:
        console.print(f"[red]Unknown source type: {source_path.suffix}[/red]")
        raise click.Abort()


def _detect_images(det, files: list[Path], out_dir: Path, console):
    """Run detection on a list of image files."""
    for f in files:
        console.print(f"Processing {f.name}...")
        result = det(str(f))
        out_path = out_dir / f.name
        result.save(out_path)
        console.print(f"  {len(result.boxes)} detections -> {out_path}")

    console.print(f"[green]Done! {len(files)} images saved to {out_dir}[/green]")


def _detect_video(det, source_path: Path, out_dir: Path, console, skip_frames: int, codec: str):
    """Run detection on a video file."""
    out_path = out_dir / source_path.name
    console.print(f"Processing video {source_path.name}...")

    stats = det.detect_video_to_file(
        source=str(source_path),
        output=str(out_path),
        codec=codec,
        skip_frames=skip_frames,
    )

    console.print(
        f"  {stats['total_frames']} frames, "
        f"{stats['processed_frames']} processed, "
        f"{stats['total_detections']} total detections"
    )
    console.print(f"[green]Done! Video saved to {out_path}[/green]")
