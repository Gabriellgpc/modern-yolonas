"""CLI: yolonas train"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--data", required=True, help="Path to dataset root.")
@click.option("--format", "data_format", default="yolo", type=click.Choice(["yolo", "coco"]))
@click.option("--epochs", default=300, help="Number of training epochs.")
@click.option("--batch-size", default=32, help="Batch size per GPU.")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--device", default="cuda", help="Device.")
@click.option("--output", default="runs/train", help="Output directory.")
@click.option("--resume", "resume_path", default=None, help="Checkpoint path to resume from.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--workers", default=8, help="DataLoader workers.")
@click.option("--pretrained/--no-pretrained", default=True, help="Use pretrained COCO weights.")
def train(
    model: str,
    data: str,
    data_format: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output: str,
    resume_path: str | None,
    input_size: int,
    workers: int,
    pretrained: bool,
):
    """Train a YOLO-NAS model."""
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, HSVAugment, HorizontalFlip, RandomAffine, LetterboxResize, Normalize
    from modern_yolonas.data.collate import detection_collate_fn
    from modern_yolonas.training.trainer import Trainer

    console = Console()

    # Build model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Building {model} (pretrained={pretrained})...")
    yolo_model = builders[model](pretrained=pretrained)

    # Build dataset
    transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=input_size),
        Normalize(),
    ])

    if data_format == "yolo":
        from modern_yolonas.data.yolo import YOLODetectionDataset
        from torch.utils.data import DataLoader

        train_dataset = YOLODetectionDataset(data, split="train", transforms=transforms, input_size=input_size)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=Compose([
            LetterboxResize(target_size=input_size), Normalize()
        ]), input_size=input_size)
    else:
        from modern_yolonas.data.coco import COCODetectionDataset
        from torch.utils.data import DataLoader

        data_path = Path(data)
        train_dataset = COCODetectionDataset(
            data_path / "images" / "train2017",
            data_path / "annotations" / "instances_train2017.json",
            transforms=transforms,
            input_size=input_size,
        )
        val_dataset = COCODetectionDataset(
            data_path / "images" / "val2017",
            data_path / "annotations" / "instances_val2017.json",
            transforms=Compose([LetterboxResize(target_size=input_size), Normalize()]),
            input_size=input_size,
        )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=detection_collate_fn, pin_memory=True,
    )

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Trainer
    trainer = Trainer(
        model=yolo_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        output_dir=output,
        device=device,
    )

    if resume_path:
        trainer.resume(resume_path)

    trainer.train()
