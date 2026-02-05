"""CLI: yolonas eval"""

from __future__ import annotations

from pathlib import Path

import click


@click.command("eval")
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", required=True, help="Path to COCO dataset root.")
@click.option("--split", default="val2017", help="Split name.")
@click.option("--batch-size", default=32, help="Batch size.")
@click.option("--device", default="cuda", help="Device.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--conf", default=0.001, help="Confidence threshold for eval.")
@click.option("--iou", default=0.65, help="NMS IoU threshold for eval.")
@click.option("--checkpoint", default=None, help="Custom checkpoint path.")
def eval_cmd(
    model: str,
    data: str,
    split: str,
    batch_size: int,
    device: str,
    input_size: int,
    conf: float,
    iou: float,
    checkpoint: str | None,
):
    """Evaluate model on COCO dataset."""
    import torch
    from rich.console import Console
    from torch.utils.data import DataLoader

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.coco import COCODetectionDataset
    from modern_yolonas.data.transforms import Compose, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.inference.postprocess import postprocess
    from modern_yolonas.training.metrics import COCOEvaluator

    console = Console()
    data_path = Path(data)

    # Load model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=True)

    yolo_model = yolo_model.to(device).eval()

    # Dataset
    ann_file = data_path / "annotations" / f"instances_{split}.json"
    img_dir = data_path / "images" / split

    dataset = COCODetectionDataset(
        img_dir, ann_file,
        transforms=Compose([LetterboxResize(target_size=input_size), Normalize()]),
        input_size=input_size,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    evaluator = COCOEvaluator(ann_file)

    console.print(f"Evaluating {model} on {split} ({len(dataset)} images)...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            pred_bboxes, pred_scores = yolo_model(images)

            results = postprocess(pred_bboxes, pred_scores, conf, iou)

            # Get image IDs for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            image_ids = [dataset.ids[i] for i in range(start_idx, end_idx)]

            boxes_list = [r[0] for r in results]
            scores_list = [r[1] for r in results]
            class_ids_list = [r[2] for r in results]

            evaluator.update(image_ids, boxes_list, scores_list, class_ids_list)

            if (batch_idx + 1) % 20 == 0:
                console.print(f"  [{batch_idx + 1}/{len(loader)}]")

    metrics = evaluator.evaluate()
    console.print("\n[bold]Results:[/bold]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")
