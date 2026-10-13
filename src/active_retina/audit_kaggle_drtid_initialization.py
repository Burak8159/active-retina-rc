"""Development-only baseline audit for the released Kaggle DR ResNet-50."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_drtid_paired_fusion import (
    PairedFundusDataset,
    PairedFusionModel,
    patient_split,
    read_pairs,
    referable_score,
    safe_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path.cwd()
    manifest = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    pairs = read_pairs(manifest, root)
    _train_pairs, valid_pairs = patient_split(pairs, 0.2, args.seed)

    import timm

    model = PairedFusionModel(
        "resnet50.a1_in1k", pretrained=False, unfreeze_layer4=False,
        architecture="pooled", anchored_fusion=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("net", checkpoint)
    state = {key.removeprefix("module."): value for key, value in source_state.items()}
    fc_weight, fc_bias = state.pop("fc.weight"), state.pop("fc.bias")
    incompatible = model.encoder.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint mapping: {incompatible}")
    with torch.no_grad():
        for head in model.view_heads:
            head.weight.copy_(fc_weight)
            head.bias.copy_(fc_bias)

    cfg = timm.data.resolve_model_data_config(model.encoder)
    cfg["input_size"] = (3, args.image_size, args.image_size)
    cfg["mean"] = (0.372487, 0.217266, 0.119367)
    cfg["std"] = (0.281526, 0.179457, 0.109162)
    transform = timm.data.create_transform(**cfg, is_training=False)
    loader = DataLoader(
        PairedFundusDataset(valid_pairs, transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    model.to(args.device).eval()
    fused_scores = np.empty(len(valid_pairs), dtype=np.float64)
    view_scores = np.empty((len(valid_pairs), 2), dtype=np.float64)
    labels = np.empty(len(valid_pairs), dtype=np.int64)
    grades = np.empty(len(valid_pairs), dtype=np.int64)
    with torch.inference_mode():
        for images, grade, label, indices in loader:
            fused, per_view, _quality = model(images.to(args.device, non_blocking=True))
            i = indices.numpy()
            fused_scores[i] = referable_score(fused).cpu().numpy()
            view_scores[i] = torch.stack(
                [referable_score(per_view[:, j]) for j in range(2)], dim=1
            ).cpu().numpy()
            labels[i] = label.numpy()
            grades[i] = grade.numpy()

    outputs = {
        "status": "DEVELOPMENT_ONLY_PRETRAINED_INITIALIZATION_AUDIT",
        "split": "patient-disjoint 80/20 split of official DRTiD training partition; official testing rows excluded",
        "seed": args.seed,
        "n_train": len(pairs) - len(valid_pairs),
        "n_validation": len(valid_pairs),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "image_size": args.image_size,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "metrics": {
            "paired_mean_logits": safe_metrics(labels, fused_scores),
            "macula": safe_metrics(labels, view_scores[:, 0]),
            "optic_disc": safe_metrics(labels, view_scores[:, 1]),
        },
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", encoding="utf-8") as stream:
        stream.write("eye_id,grade,referable_dr,paired_score,macula_score,optic_disc_score\n")
        for i, pair in enumerate(valid_pairs):
            stream.write(
                f"{pair['eye_id']},{grades[i]},{labels[i]},{fused_scores[i]:.9g},"
                f"{view_scores[i, 0]:.9g},{view_scores[i, 1]:.9g}\n"
            )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
