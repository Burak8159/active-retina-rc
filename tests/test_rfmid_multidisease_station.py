import numpy as np
import pytest
import torch
from PIL import Image

from active_retina.rfmid_multidisease_station import decode_image, make_response


def test_outputs_are_per_condition_and_risk_is_separate():
    labels = ["Disease_Risk", "DR", "ARMD"]
    bundle = {
        "labels": labels,
        "per_label_reference": {"DR": {"n": 640, "positive": 124, "sensitivity_at_0_5": 0.9}},
        "architecture": "torchvision.convnext_tiny",
        "epoch": 20,
        "checkpoint_sha256": "abc",
        "device": "cpu",
    }
    result = make_response(bundle, [0.7, 0.8, 0.2], "case-1")
    assert result["dataset_overall_risk_label"]["output"] == "FLAGGED"
    assert [x["label"] for x in result["conditions"]] == ["DR", "ARMD"]
    assert [x["output"] for x in result["conditions"]] == ["FLAGGED", "NOT_FLAGGED"]
    assert result["condition_outputs_are_independent"] is True
    assert result["autonomous_diagnosis"] is False
    assert result["image_saved"] is False


def test_decode_rejects_non_image_bytes():
    with pytest.raises(ValueError, match="not a readable"):
        decode_image(b"not an image")


def test_decode_accepts_image_bytes():
    import io
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (16, 16)
