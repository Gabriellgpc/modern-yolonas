"""Tests for model architecture: forward pass shapes, train vs eval mode."""

import pytest
import torch

from modern_yolonas import yolo_nas_s, yolo_nas_m, yolo_nas_l
from modern_yolonas.model import YoloNAS


@pytest.fixture(params=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
def variant(request):
    return request.param


@pytest.fixture
def model(variant):
    builders = {"yolo_nas_s": yolo_nas_s, "yolo_nas_m": yolo_nas_m, "yolo_nas_l": yolo_nas_l}
    return builders[variant](pretrained=False)


class TestForwardPass:
    def test_eval_mode_output_shapes(self, model):
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        pred_bboxes, pred_scores = model(x)
        # 640/8=80, 640/16=40, 640/32=20 → 80*80 + 40*40 + 20*20 = 8400
        assert pred_bboxes.shape == (1, 8400, 4)
        assert pred_scores.shape == (1, 8400, 80)

    def test_eval_mode_batch(self, model):
        model.eval()
        x = torch.randn(2, 3, 640, 640)
        pred_bboxes, pred_scores = model(x)
        assert pred_bboxes.shape == (2, 8400, 4)
        assert pred_scores.shape == (2, 8400, 80)

    def test_train_mode_returns_raw_predictions(self, model):
        model.train()
        x = torch.randn(1, 3, 640, 640)
        out = model(x)
        decoded, raw = out
        pred_bboxes, pred_scores = decoded
        assert pred_bboxes.shape == (1, 8400, 4)
        assert pred_scores.shape == (1, 8400, 80)
        # Raw predictions: cls_logits, reg_distri, anchors, anchor_points, num_anchors_list, stride_tensor
        cls_logits, reg_distri, anchors, anchor_points, num_anchors_list, stride_tensor = raw
        assert cls_logits.shape == (1, 8400, 80)

    def test_different_input_sizes(self, model):
        model.eval()
        for size in [320, 480]:
            x = torch.randn(1, 3, size, size)
            pred_bboxes, pred_scores = model(x)
            expected = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
            assert pred_bboxes.shape == (1, expected, 4)


class TestModelConstruction:
    def test_from_config(self):
        model = YoloNAS.from_config("yolo_nas_s", num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        _, pred_scores = model(x)
        assert pred_scores.shape[-1] == 10

    def test_custom_num_classes(self):
        for nc in [1, 10, 365]:
            model = YoloNAS.from_config("yolo_nas_s", num_classes=nc)
            model.eval()
            x = torch.randn(1, 3, 640, 640)
            _, pred_scores = model(x)
            assert pred_scores.shape[-1] == nc
