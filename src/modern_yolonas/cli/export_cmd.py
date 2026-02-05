"""CLI: yolonas export"""

from __future__ import annotations

import click


@click.command()
@click.option("--model", default="yolo_nas_s", type=click.Choice(["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]))
@click.option("--format", "export_format", default="onnx", type=click.Choice(["onnx", "openvino"]))
@click.option("--output", default=None, help="Output file path (default: model.onnx or model.xml).")
@click.option("--input-size", default=640, help="Model input size.")
@click.option("--opset", default=17, help="ONNX opset version.")
@click.option("--checkpoint", default=None, help="Custom checkpoint path.")
def export(model: str, export_format: str, output: str | None, input_size: int, opset: int, checkpoint: str | None):
    """Export model to ONNX or OpenVINO format."""
    import torch
    from rich.console import Console

    from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l

    console = Console()

    if output is None:
        output = "model.xml" if export_format == "openvino" else "model.onnx"

    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    console.print(f"Loading {model}...")

    if checkpoint:
        yolo_model = builders[model](pretrained=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        yolo_model.load_state_dict(sd)
    else:
        yolo_model = builders[model](pretrained=True)

    yolo_model.eval()

    # Fuse RepVGG blocks for deployment
    for module in yolo_model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()

    dummy = torch.randn(1, 3, input_size, input_size)

    if export_format == "openvino":
        import openvino as ov

        console.print("Exporting to OpenVINO IR...")
        ov_model = ov.convert_model(yolo_model, example_input=dummy)
        ov.save_model(ov_model, output)
    else:
        console.print(f"Exporting to ONNX (opset {opset})...")
        torch.onnx.export(
            yolo_model,
            dummy,
            output,
            input_names=["images"],
            output_names=["pred_bboxes", "pred_scores"],
            dynamic_axes={
                "images": {0: "batch"},
                "pred_bboxes": {0: "batch"},
                "pred_scores": {0: "batch"},
            },
            opset_version=opset,
        )

    console.print(f"[green]Exported to {output}[/green]")
