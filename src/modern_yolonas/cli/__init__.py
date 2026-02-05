import click

from modern_yolonas.cli.detect_cmd import detect
from modern_yolonas.cli.train_cmd import train
from modern_yolonas.cli.export_cmd import export
from modern_yolonas.cli.eval_cmd import eval_cmd


@click.group()
@click.version_option(package_name="modern-yolonas")
def main():
    """YOLO-NAS object detection."""


main.add_command(detect)
main.add_command(train)
main.add_command(export)
main.add_command(eval_cmd, name="eval")
