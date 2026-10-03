from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderSpec:
    family: str
    candidate: str
    fallback: str
    pretrained: bool
    image_size: int


def build_encoder(spec: EncoderSpec):
    try:
        import timm
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Training requires torch and timm; evaluation from exported predictions does not.") from exc

    name = spec.candidate
    try:
        model = timm.create_model(name, pretrained=spec.pretrained, num_classes=0)
    except Exception:
        name = spec.fallback
        model = timm.create_model(name, pretrained=spec.pretrained, num_classes=0)
    dim = getattr(model, "num_features", None)
    if dim is None:
        raise RuntimeError(f"Could not infer feature dimension for encoder {name}")
    return nn.Sequential(model), int(dim), name


def build_set_classifier(feature_dim: int, hidden_dim: int, dropout: float, n_classes: int = 5):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Set pooling requires torch.") from exc

    class SetPoolingClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(feature_dim * 2 + 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        def forward(self, embeddings, mask):
            masked = embeddings * mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            mean_pool = masked.sum(dim=1) / denom
            very_neg = torch.finfo(embeddings.dtype).min
            max_pool = embeddings.masked_fill(mask.unsqueeze(-1) == 0, very_neg).max(dim=1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
            features = torch.cat([mean_pool, max_pool, mask], dim=-1)
            return self.head(features)

    return SetPoolingClassifier()
