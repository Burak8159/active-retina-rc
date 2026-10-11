"""Audit conservative late fusion of paired-view model scores on development data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from active_retina.train_mixed_view_bag_autonomy import EyeBags, MixedViewModel
from active_retina.train_quality_gated_autonomy import evaluate_policy, read_rows, select_policy


def collect(model: MixedViewModel, data: EyeBags, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    values: dict[str, list[Any]] = {key: [] for key in (
        "grade", "quality_known", "ungradable", "fused", "view0", "view1", "quality",
        "eye_id", "patient_id", "source_dataset", "image_ids",
    )}
    model.eval()
    with torch.inference_mode():
        for images, mask, grade, _referable, quality_known, ungradable, eye_id, patient_id, source, image_ids in loader:
            images, mask = images.to(device), mask.to(device)
            _, fused_risk, fused_quality = model(images, mask)
            first_mask = torch.zeros_like(mask)
            first_mask[:, 0] = mask[:, 0]
            second_mask = torch.zeros_like(mask)
            second_mask[:, 1] = mask[:, 1]
            missing_second = ~mask[:, 1]
            second_mask[missing_second, 0] = mask[missing_second, 0]
            _, first_risk, _ = model(images, first_mask)
            _, second_risk, _ = model(images, second_mask)
            batch_values = {
                "grade": grade.numpy(), "quality_known": quality_known.numpy(),
                "ungradable": ungradable.numpy(), "fused": fused_risk.float().sigmoid().cpu().numpy(),
                "view0": first_risk.float().sigmoid().cpu().numpy(),
                "view1": second_risk.float().sigmoid().cpu().numpy(),
                "quality": fused_quality.float().sigmoid().cpu().numpy(),
                "eye_id": eye_id, "patient_id": patient_id, "source_dataset": source, "image_ids": image_ids,
            }
            for key, batch in batch_values.items():
                values[key].extend(batch.tolist() if isinstance(batch, np.ndarray) else list(batch))
    return {key: np.asarray(value) for key, value in values.items()}


def make_policy_input(scores: dict[str, np.ndarray], rule: str) -> dict[str, np.ndarray]:
    if rule == "fused":
        risk = scores["fused"]
    elif rule == "max_any_view":
        risk = np.maximum.reduce((scores["fused"], scores["view0"], scores["view1"]))
    elif rule == "mean_fused_views":
        risk = (scores["fused"] + scores["view0"] + scores["view1"]) / 3.0
    elif rule == "max_views":
        risk = np.maximum(scores["view0"], scores["view1"])
    else:
        raise ValueError(f"Unknown score rule: {rule}")
    return {
        "grade": scores["grade"].astype(int), "risk": risk.astype(float),
        "quality": scores["quality"].astype(float),
        "quality_known": scores["quality_known"].astype(int),
        "ungradable": scores["ungradable"].astype(int),
    }


def source_metrics(pred: dict[str, np.ndarray], policy: dict[str, Any], source: np.ndarray) -> dict[str, Any]:
    output = {}
    for name in sorted(set(source.tolist())):
        indices = source == name
        subset = {key: value[indices] for key, value in pred.items()}
        output[name] = evaluate_policy(subset, policy)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MixedViewModel(str(checkpoint["model_name"]), pretrained=False, image_size=args.image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    data = {
        split: EyeBags(splits[split], model.eval_transform, root)
        for split in ("valid", "test")
    }
    scores = {split: collect(model, dataset, device, args.batch_size) for split, dataset in data.items()}

    rules = ("fused", "max_views", "max_any_view", "mean_fused_views")
    report: dict[str, Any] = {
        "status": "DEVELOPMENT_ONLY_NOT_CERTIFICATION",
        "claim_boundary": "This manifest informed prior training and model selection; all results are development evidence.",
        "unit_of_analysis": "one DDR eye or paired DRTiD eye",
        "checkpoint": args.checkpoint,
        "validation_selected_rules": {},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for rule in rules:
        val = make_policy_input(scores["valid"], rule)
        policy = select_policy(val)
        report["validation_selected_rules"][rule] = {
            "policy": policy,
            "validation_metrics": evaluate_policy(val, policy),
            "test_metrics_development_only": evaluate_policy(make_policy_input(scores["test"], rule), policy),
            "validation_source_metrics": source_metrics(val, policy, scores["valid"]["source_dataset"]),
            "test_source_metrics_development_only": source_metrics(
                make_policy_input(scores["test"], rule), policy, scores["test"]["source_dataset"]
            ),
        }
        rows.append({
            "rule": rule,
            "validation_specificity": policy["specificity_gradable_nonreferable"],
            "validation_misses": policy["missed_referable_dr"],
            "validation_referral_rate": policy["referral_rate"],
            **{
                f"test_{key}": value
                for key, value in report["validation_selected_rules"][rule]["test_metrics_development_only"].items()
            },
        })
    report["best_validation_rule"] = max(
        rules,
        key=lambda rule: (
            report["validation_selected_rules"][rule]["validation_metrics"]["specificity_gradable_nonreferable"],
            -report["validation_selected_rules"][rule]["validation_metrics"]["referral_rate"],
        ),
    )
    with (output / "fusion_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
