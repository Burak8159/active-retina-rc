"""Validation-only audit of paired-view risk pooling and per-field signal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_mixed_view_bag_autonomy import (
    EyeBags,
    MixedViewModel,
    pool_view_risk,
    stratified_policy_metrics,
)
from .train_quality_gated_autonomy import evaluate_policy, read_rows, select_policy


def collect_per_view_logits(model: MixedViewModel, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    captured: list[torch.Tensor] = []
    handle = model.view_risk_head.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    rows: list[dict[str, Any]] = []
    try:
        model.eval()
        with torch.inference_mode():
            for images, mask, grade, referable, quality_known, ungradable, eye_id, patient_id, source, image_ids in loader:
                captured.clear()
                _grade_logits, _pooled_risk, quality_logits = model(images.to(device), mask.to(device))
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one view-risk head invocation per eye batch")
                view_logits = captured[0].squeeze(-1).float().cpu().numpy()
                valid = mask.numpy()
                for i, key in enumerate(eye_id):
                    rows.append({
                        "eye_id": str(key),
                        "patient_id": str(patient_id[i]),
                        "source_dataset": str(source[i]),
                        "grade": int(grade[i]),
                        "referable": int(referable[i]),
                        "quality_known": int(quality_known[i]),
                        "ungradable": int(ungradable[i]),
                        "quality": float(torch.sigmoid(quality_logits[i].float()).cpu()),
                        "view_logits": view_logits[i].tolist(),
                        "view_mask": valid[i].tolist(),
                    })
    finally:
        handle.remove()
    return rows


def policy_result(rows: list[dict[str, Any]], risk: np.ndarray) -> dict[str, Any]:
    scores = {
        "grade": np.asarray([r["grade"] for r in rows]),
        "risk": risk,
        "quality": np.asarray([r["quality"] for r in rows]),
        "quality_known": np.asarray([r["quality_known"] for r in rows]),
        "ungradable": np.asarray([r["ungradable"] for r in rows]),
        "source_dataset": np.asarray([r["source_dataset"] for r in rows]),
    }
    policy = select_policy(scores)
    return {
        "policy": policy,
        "metrics": evaluate_policy(scores, policy),
        "metrics_by_source_and_grade": stratified_policy_metrics(scores, policy),
    }


def paired_field_results(rows: list[dict[str, Any]], field_by_eye: dict[str, list[str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("macula", "optic_disc"):
        selected: list[dict[str, Any]] = []
        scores: list[float] = []
        for row in rows:
            fields = field_by_eye[row["eye_id"]]
            if row["source_dataset"] not in {"drtid_paired", "deepdrid_paired"} or len(fields) != 2:
                continue
            index = fields.index(field)
            if not row["view_mask"][index]:
                continue
            selected.append({**row, "quality_known": 0, "ungradable": 0, "quality": 0.0})
            scores.append(float(torch.sigmoid(torch.tensor(row["view_logits"][index])).item()))
        output[field] = policy_result(selected, np.asarray(scores, dtype=float))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {})
    model = MixedViewModel(
        args.model_name,
        pretrained=False,
        image_size=args.image_size,
        preserve_pretrained_risk=bool(saved_args.get("preserve_pretrained_risk", False)),
        risk_pooling=str(saved_args.get("risk_pooling", "logmeanexp")),
        risk_pool_temperature=float(saved_args.get("risk_pool_temperature", 0.25)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)

    root = Path.cwd()
    valid_rows = read_rows(Path(args.manifest))["valid"]
    dataset = EyeBags(valid_rows, model.eval_transform, root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    fields = {item["eye_id"]: [view[0] for view in item["views"]] for item in dataset.items}
    view_rows = collect_per_view_logits(model, loader, device)
    if len(view_rows) != len(dataset):
        raise RuntimeError("Prediction count did not match the validation-eye count")

    pooled: dict[str, Any] = {}
    for name, mode, temperature in (
        ("logmeanexp_t0.10", "logmeanexp", 0.10),
        ("logmeanexp_t0.25", "logmeanexp", 0.25),
        ("logmeanexp_t0.50", "logmeanexp", 0.50),
        ("logmeanexp_t1.00", "logmeanexp", 1.00),
        ("max", "max", 0.25),
        ("mean", "mean", 0.25),
    ):
        logits = torch.tensor([row["view_logits"] for row in view_rows], dtype=torch.float32)
        mask = torch.tensor([row["view_mask"] for row in view_rows], dtype=torch.bool)
        pooled_logits = pool_view_risk(logits, mask, mode, temperature)
        pooled[name] = policy_result(view_rows, torch.sigmoid(pooled_logits).numpy())

    report = {
        "status": "VALIDATION_DIAGNOSTIC_ONLY",
        "claim_boundary": "The validation split has informed prior model and policy development; no test images were loaded.",
        "checkpoint": args.checkpoint,
        "n_validation_eyes": len(view_rows),
        "paired_view_field_metrics": paired_field_results(view_rows, fields),
        "pooling_sensitivity": pooled,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "n_validation_eyes": len(view_rows),
        "paired_fields": {key: value["metrics"] for key, value in report["paired_view_field_metrics"].items()},
        "pooling": {key: value["metrics"] for key, value in report["pooling_sensitivity"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
