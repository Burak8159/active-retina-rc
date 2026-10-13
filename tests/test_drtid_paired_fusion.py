import pytest
import torch

pytest.importorskip("timm")

from active_retina.train_drtid_paired_fusion import PairedFusionModel


@pytest.mark.parametrize("architecture", ["pooled", "cross_attention"])
def test_paired_fusion_model_output_shapes(architecture):
    model = PairedFusionModel(
        "resnet50.a1_in1k",
        pretrained=False,
        unfreeze_layer4=False,
        architecture=architecture,
    ).eval()
    with torch.no_grad():
        fused, per_view, quality = model(torch.randn(1, 2, 3, 224, 224))

    assert fused.shape == (1, 5)
    assert per_view.shape == (1, 2, 5)
    assert quality.shape == (1, 2)
    assert torch.isfinite(fused).all()


def test_anchored_fusion_starts_at_mean_pretrained_view_logits():
    model = PairedFusionModel(
        "resnet50.a1_in1k",
        pretrained=False,
        unfreeze_layer4=False,
        architecture="pooled",
        anchored_fusion=True,
    ).eval()
    with torch.no_grad():
        fused, per_view, _quality = model(torch.randn(1, 2, 3, 224, 224))

    torch.testing.assert_close(fused, per_view.mean(dim=1), rtol=0, atol=0)
