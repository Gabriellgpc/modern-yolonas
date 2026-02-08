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
@click.option("--output", default="runs/train", help="Output directory.")
@click.option("--resume", "resume_path", default=None, help="Checkpoint path to resume from.")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--workers", default=8, help="DataLoader workers.")
@click.option("--pretrained/--no-pretrained", default=True, help="Use pretrained COCO weights.")
@click.option("--num-classes", default=80, help="Number of object classes.")
@click.option("--devices", default="auto", help="Devices to use (e.g. 'auto', '1', '0,1').")
@click.option(
    "--logger",
    default="csv",
    type=click.Choice(["csv", "tensorboard", "wandb"]),
    help="Logger backend.",
)
def train(
    model: str,
    data: str,
    data_format: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output: str,
    resume_path: str | None,
    input_size: int,
    workers: int,
    pretrained: bool,
    num_classes: int,
    devices: str,
    logger: str,
):
    """Train a YOLO-NAS model."""
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
    from modern_yolonas.data.transforms import Compose, HSVAugment, HorizontalFlip, RandomAffine, LetterboxResize, Normalize
    from modern_yolonas.training import YoloNASLightningModule, EMACallback, DetectionDataModule

    console = Console()

    # Build model
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Building {model} (pretrained={pretrained}, num_classes={num_classes})...")
    yolo_model = builders[model](pretrained=pretrained, num_classes=num_classes)

    # Build datasets
    transforms = Compose([
        HSVAugment(),
        HorizontalFlip(),
        RandomAffine(degrees=0.0, translate=0.1, scale=(0.5, 1.5)),
        LetterboxResize(target_size=input_size),
        Normalize(),
    ])

    if data_format == "yolo":
        from modern_yolonas.data.yolo import YOLODetectionDataset

        train_dataset = YOLODetectionDataset(data, split="train", transforms=transforms, input_size=input_size)
        val_dataset = YOLODetectionDataset(data, split="val", transforms=Compose([
            LetterboxResize(target_size=input_size), Normalize()
        ]), input_size=input_size)
    else:
        from modern_yolonas.data.coco import COCODetectionDataset

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

    console.print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    # Lightning components
    warmup_steps = min(1000, len(train_dataset) // batch_size * 3)
    lit_model = YoloNASLightningModule(
        model=yolo_model,
        num_classes=num_classes,
        lr=lr,
        warmup_steps=warmup_steps,
    )

    data_module = DetectionDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    # Logger
    if logger == "csv":
        logger_instance = L.pytorch.loggers.CSVLogger(output)
    elif logger == "tensorboard":
        logger_instance = L.pytorch.loggers.TensorBoardLogger(output)
    else:
        logger_instance = L.pytorch.loggers.WandbLogger(project="yolonas", save_dir=output)

    # Parse devices
    parsed_devices: str | int | list[int] = devices
    if devices != "auto":
        if "," in devices:
            parsed_devices = [int(d) for d in devices.split(",")]
        else:
            parsed_devices = int(devices)

    callbacks = [
        EMACallback(),
        ModelCheckpoint(dirpath=output, save_last=True),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=parsed_devices,
        strategy="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger_instance,
        default_root_dir=output,
    )

    if resume_path:
        trainer.fit(lit_model, datamodule=data_module, ckpt_path=resume_path)
    else:
        trainer.fit(lit_model, datamodule=data_module)
