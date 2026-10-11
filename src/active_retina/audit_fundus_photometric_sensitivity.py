"""Development-only photometric sensitivity audit for a frozen paired-view model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from active_retina.train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    predictions,
    select_filtered_policy,
)
from active_retina.train_quality_gated_autonomy import evaluate_policy, read_rows


def photometric_image(image: Image.Image, mode: str) -> Image.Image:
    """Apply a fixed, label-independent transform before the model's eval transform."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mode == "identity":
        return Image.fromarray(rgb.astype(np.uint8))
    if mode == "green_channel":
        green = rgb[:, :, 1].astype(np.uint8)
        return Image.fromarray(np.repeat(green[:, :, None], 3, axis=2))
    if mode == "gray_world":
        means = rgb.mean(axis=(0, 1))
        target = float(np.exp(np.log(np.maximum(means, 1.0)).mean()))
        balanced = rgb * (target / np.maximum(means, 1.0))[None, None, :]
        return Image.fromarray(np.clip(balanced, 0, 255).astype(np.uint8))
    if mode == "channel_percentile":
        normalized = np.empty_like(rgb)
        for channel in range(3):
            plane = rgb[:, :, channel]
            low, high = np.percentile(plane, (1, 99))
            if high <= low:
                normalized[:, :, channel] = plane
            else:
                normalized[:, :, channel] = (plane - low) * (255.0 / (high - low))
        return Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8))
    if mode == "green_autocontrast":
        green = Image.fromarray(rgb[:, :, 1].astype(np.uint8))
        adjusted = ImageOps.autocontrast(green, cutoff=1)
        return Image.merge("RGB", (adjusted, adjusted, adjusted))
    raise ValueError(f"Unknown photometric mode: {mode}")


class PhotometricTransform:
    def __init__(self, base: Callable[[Image.Image], torch.Tensor], mode: str) -> None:
        self.base = base
        self.mode = mode

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.base(photometric_image(image, self.mode))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default="drtid_paired")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    rows = [row for row in splits["valid"] if row.get("source_dataset") == args.source]
    if not rows:
        raise ValueError(f"No validation rows for source {args.source}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = MixedViewModel(
        model_name=checkpoint["model_name"],
        pretrained=False,
        image_size=int(saved_args.get("image_size", 224)),
        preserve_pretrained_risk=bool(saved_args.get("preserve_pretrained_risk", False)),
        risk_pooling=str(saved_args.get("risk_pooling", "fused")),
        risk_pool_temperature=float(saved_args.get("risk_pool_temperature", 0.25)),
        lesion_presence_classes=0,
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_prefixes = ("view_risk_head.", "view_attention_head.", "view_residual_head.", "lesion_head.")
    if incompatible.unexpected_keys or any(
        not name.startswith(allowed_prefixes) for name in incompatible.missing_keys
    ):
        raise ValueError(
            f"Checkpoint/model mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device).eval()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modes = ("identity", "green_channel", "gray_world", "channel_percentile", "green_autocontrast")
    reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for mode in modes:
        dataset = EyeBags(rows, PhotometricTransform(model.eval_transform, mode), root)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        scores = predictions(model, loader, device)
        filtered, policy = select_filtered_policy(scores, [args.source], "risk")
        metrics = evaluate_policy(
            {
                "grade": filtered["grade"],
                "risk": filtered["risk"],
                "quality": filtered["quality"],
                "quality_known": filtered["quality_known"],
                "ungradable": filtered["ungradable"],
            },
            policy,
        )
        reports.append({"mode": mode, "policy": policy, "metrics": metrics})
        for index in range(len(filtered["grade"])):
            prediction_rows.append({
                "mode": mode,
                "source_dataset": str(filtered["source_dataset"][index]),
                "eye_id": str(filtered["eye_id"][index]),
                "patient_id": str(filtered["patient_id"][index]),
                "grade": int(filtered["grade"][index]),
                "risk": float(filtered["risk"][index]),
                "quality": float(filtered["quality"][index]),
                "action": "REFER" if float(filtered["risk"][index]) >= float(policy["risk_threshold"]) else "CLEAR",
            })
    with (output / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary = {
        "status": "DEVELOPMENT_ONLY_PHOTOMETRIC_SENSITIVITY_AUDIT",
        "source": args.source,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "n_validation_eyes": len(rows) // 2,
        "modes": reports,
        "claim_boundary": "Every mode and its zero-miss threshold are assessed on the same previously inspected validation cohort; exploratory only, not independent confirmation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
