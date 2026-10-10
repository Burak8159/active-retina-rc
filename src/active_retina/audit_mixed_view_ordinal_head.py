"""Compare the ordinal grade head with the separate risk head on fixed splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from active_retina.train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    write_predictions,
)
from active_retina.train_quality_gated_autonomy import evaluate_policy, read_rows, select_policy


def ordinal_referable_probability(grade_logits: torch.Tensor) -> torch.Tensor:
    if grade_logits.ndim != 2 or grade_logits.shape[1] != 5:
        raise ValueError("Expected five ordinal grade logits")
    return grade_logits.float().softmax(dim=1)[:, 2:].sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model_name"])
    model = MixedViewModel(model_name, pretrained=False, image_size=args.image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    datasets = {
        name: EyeBags(rows, model.eval_transform, root)
        for name, rows in (("valid", splits["valid"]), ("test", splits["test"]))
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=device.type == "cuda")
        for name, dataset in datasets.items()
    }

    scores: dict[str, dict[str, np.ndarray]] = {}
    for split, loader in loaders.items():
        model.eval()
        columns: dict[str, list[Any]] = {key: [] for key in (
            "grade", "referable", "quality_known", "ungradable", "risk", "quality",
            "eye_id", "patient_id", "source_dataset", "image_ids",
        )}
        with torch.inference_mode():
            for images, mask, grade, referable, quality_known, ungradable, eye_id, patient_id, source, image_ids in loader:
                grade_logits, _, quality_logits = model(images.to(device), mask.to(device))
                risk = ordinal_referable_probability(grade_logits).cpu().numpy()
                values = {
                    "grade": grade.numpy(), "referable": referable.numpy(),
                    "quality_known": quality_known.numpy(), "ungradable": ungradable.numpy(),
                    "risk": risk, "quality": quality_logits.float().sigmoid().cpu().numpy(),
                    "eye_id": eye_id, "patient_id": patient_id,
                    "source_dataset": source, "image_ids": image_ids,
                }
                for key, value in values.items():
                    columns[key].extend(value.tolist() if isinstance(value, np.ndarray) else list(value))
        scores[split] = {key: np.asarray(value) for key, value in columns.items()}

    policy_input = {key: scores["valid"][key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    policy = select_policy(policy_input)
    summary = {
        "status": "DEVELOPMENT_DIAGNOSTIC_NOT_CERTIFIED",
        "score": "ordinal softmax probability mass for grades 2-4",
        "checkpoint": args.checkpoint,
        "backbone": model_name,
        "manifest": args.manifest,
        "validation_policy": policy,
        "validation_metrics": evaluate_policy(policy_input, policy),
        "test_metrics_development_only": evaluate_policy(
            {key: scores["test"][key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")},
            policy,
        ),
        "claim_boundary": "Splits were used during prior development; test metrics are not untouched certification evidence.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_predictions(out / "validation_predictions.csv", "valid", scores["valid"], policy)
    write_predictions(out / "test_predictions_development_only.csv", "test", scores["test"], policy)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
