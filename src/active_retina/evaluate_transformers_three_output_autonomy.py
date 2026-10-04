"""Evaluate a Transformers DR classifier under the three-output autonomy contract.

The contract is product-facing rather than benchmark-only: every image receives
one autonomous output, but only nonreferable/referable outputs are counted as
diagnostic decisions. Insufficient-evidence/reacquire is a completed safety
action, not a disease label.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from active_retina.three_output_autonomous_triage import image_quality, quality_flag, resolve_image_path, triage_metrics


def read_manifest(path: Path, max_rows_per_split: int, max_rows_per_grade_per_split: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    grade_counts: dict[tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split = row["source_split"]
            grade = int(row["grade"])
            if max_rows_per_split > 0 and split_counts.get(split, 0) >= max_rows_per_split:
                continue
            grade_key = (split, grade)
            if max_rows_per_grade_per_split > 0 and grade_counts.get(grade_key, 0) >= max_rows_per_grade_per_split:
                continue
            split_counts[split] = split_counts.get(split, 0) + 1
            grade_counts[grade_key] = grade_counts.get(grade_key, 0) + 1
            row["grade"] = grade
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


class FundusRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path, grayscale_triplicate: bool) -> None:
        self.rows = rows
        self.root = root
        self.grayscale_triplicate = grayscale_triplicate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        path = resolve_image_path(str(row["image_path"]), self.root)
        with Image.open(path) as img:
            image = img.convert("L").convert("RGB") if self.grayscale_triplicate else img.convert("RGB")
        return {
            "image": image,
            "image_id": str(row["image_id"]),
            "grade": int(row["grade"]),
            "referable_dr": int(row["referable_dr"]),
            "path": str(path),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [x["image"] for x in batch],
        "image_ids": [x["image_id"] for x in batch],
        "grades": np.asarray([x["grade"] for x in batch], dtype=int),
        "referable_dr": np.asarray([x["referable_dr"] for x in batch], dtype=int),
        "paths": [x["path"] for x in batch],
    }


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def infer_positive_indices(config: Any, explicit: str) -> list[int]:
    if explicit.strip():
        return [int(x) for x in explicit.split(",") if x.strip()]
    id2label = getattr(config, "id2label", None) or {}
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    out = [
        idx
        for idx, text in labels.items()
        if any(token in text for token in ("moderate", "severe", "proliferative", "referable", "grade_2", "grade_3", "grade_4"))
    ]
    if out:
        return sorted(out)
    n = int(getattr(config, "num_labels", 5))
    if n >= 5:
        return list(range(2, n))
    if n == 2:
        return [1]
    return [n - 1]


def model_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)):
        logits = output[0]
    if logits is None:
        raise RuntimeError("Transformers model did not return logits; this evaluator needs a classification head.")
    return logits


def predict_split(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    root: Path,
    batch_size: int,
    device: torch.device,
    positive_indices: list[int],
    use_amp: bool,
    grayscale_triplicate: bool,
    invert_score: bool,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        FundusRows(rows, root, grayscale_triplicate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    out: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = processor(images=batch["images"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
                logits = model_logits(model(**inputs))
                if logits.ndim == 1 or logits.shape[1] == 1:
                    scores = torch.sigmoid(logits.reshape(-1))
                else:
                    probs = torch.softmax(logits, dim=1)
                    safe_indices = [idx for idx in positive_indices if idx < probs.shape[1]]
                    if not safe_indices:
                        safe_indices = [probs.shape[1] - 1]
                    scores = probs[:, safe_indices].sum(dim=1)
            for image_id, grade, y, score, path in zip(
                batch["image_ids"],
                batch["grades"],
                batch["referable_dr"],
                scores.detach().cpu().float().numpy(),
                batch["paths"],
            ):
                final_score = 1.0 - float(score) if invert_score else float(score)
                q = image_quality(Path(path))
                out.append(
                    {
                        "image_id": str(image_id),
                        "grade": int(grade),
                        "referable_dr": int(y),
                        "score": final_score,
                        "quality_ok": bool(quality_flag(q)),
                        **q,
                    }
                )
    return out


def action(score: float, low: float, high: float, quality_ok: bool) -> str:
    if not quality_ok:
        return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"
    if score <= low:
        return "AUTONOMOUS_NONREFERABLE"
    if score >= high:
        return "AUTONOMOUS_REFERABLE"
    return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"


def apply_policy(rows: list[dict[str, Any]], low: float, high: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["low_threshold"] = float(low)
        item["high_threshold"] = float(high)
        item["action"] = action(float(row["score"]), float(low), float(high), bool(row["quality_ok"]))
        out.append(item)
    return out


def select_policy(valid: list[dict[str, Any]], min_specificity: float) -> dict[str, Any]:
    scores = sorted({float(r["score"]) for r in valid})
    if not scores:
        return {"status": "NO_SCORES"}
    quantiles = np.quantile(scores, np.linspace(0, 1, 121)).tolist()
    grid = sorted(set([scores[0] - 1e-9, scores[-1] + 1e-9, *quantiles]))
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            pred = apply_policy(valid, float(low), float(high))
            metrics = triage_metrics(pred)
            if metrics["false_negative_clearances"] != 0:
                continue
            if metrics["specificity_among_diagnostic_decisions"] < min_specificity:
                continue
            key = (
                float(metrics["diagnostic_decision_rate"]),
                float(metrics["negative_clearance_rate"]),
                float(metrics["referable_autonomous_referral_rate"]),
            )
            candidate = {"low": float(low), "high": float(high), **metrics}
            if best is None or key > best[0]:
                best = (key, candidate)
    return best[1] if best else {"status": "NO_VALID_THREE_OUTPUT_POLICY"}


def write_predictions(path: Path, split: str, rows: list[dict[str, Any]]) -> None:
    exists = path.exists()
    fields = [
        "split",
        "image_id",
        "grade",
        "referable_dr",
        "score",
        "low_threshold",
        "high_threshold",
        "quality_ok",
        "mean_luminance",
        "contrast",
        "edge_energy",
        "dark_fraction",
        "bright_fraction",
        "action",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"split": split, **{k: row.get(k, "") for k in fields if k != "split"}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="results/ddr_hf_specificity300_manifest.csv")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-indices", default="2,3,4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grayscale-triplicate", action="store_true")
    parser.add_argument("--invert-score", action="store_true")
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--baseline-summary", default="results/three_output_autonomous_triage/summary.json")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--max-rows-per-grade-per-split", type=int, default=0)
    parser.add_argument("--prediction-splits", default="valid,test")
    args = parser.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    rows = read_manifest(Path(args.manifest), args.max_rows_per_split, args.max_rows_per_grade_per_split)
    by_split = split_rows(rows)
    device = device_for(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id).to(device)
    positive_indices = infer_positive_indices(model.config, args.positive_indices)

    requested_splits = [x.strip() for x in args.prediction_splits.split(",") if x.strip()]
    raw_predictions: dict[str, list[dict[str, Any]]] = {}
    for split in requested_splits:
        raw_predictions[split] = predict_split(
            model,
            processor,
            by_split[split],
            root,
            args.batch_size,
            device,
            positive_indices,
            args.use_amp,
            args.grayscale_triplicate,
            args.invert_score,
        )

    if "valid" not in raw_predictions or "test" not in raw_predictions:
        pred_path = out / "predictions.csv"
        if pred_path.exists():
            pred_path.unlink()
        for split in requested_splits:
            write_predictions(pred_path, split, raw_predictions[split])
        summary = {
            "status": "PREDICTIONS_ONLY",
            "model_id": args.model_id,
            "config_id2label": getattr(model.config, "id2label", None),
            "grayscale_triplicate": args.grayscale_triplicate,
            "invert_score": args.invert_score,
            "positive_indices": positive_indices,
            "prediction_splits": requested_splits,
            "predictions": str(pred_path),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    valid = raw_predictions["valid"]
    selected = select_policy(valid, args.min_specificity)
    low = float(selected.get("low", 0.0))
    high = float(selected.get("high", 1.0))
    valid_pred = apply_policy(valid, low, high)
    test = raw_predictions["test"]
    test_pred = apply_policy(test, low, high)

    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    for split in requested_splits:
        pred_rows = apply_policy(raw_predictions[split], low, high) if split in ("valid", "test") else raw_predictions[split]
        write_predictions(pred_path, split, pred_rows)
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    baseline_test = baseline["test_triage_metrics"]
    test_metrics = triage_metrics(test_pred)
    improved = (
        test_metrics["false_negative_clearances"] == 0
        and test_metrics["specificity_among_diagnostic_decisions"] >= args.min_specificity
        and test_metrics["diagnostic_decision_rate"] > float(baseline_test["diagnostic_decision_rate"])
    )
    summary = {
        "status": "GO_IMPROVED_THREE_OUTPUT_AUTONOMY" if improved else "NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY",
        "model_id": args.model_id,
        "config_id2label": getattr(model.config, "id2label", None),
        "grayscale_triplicate": args.grayscale_triplicate,
        "invert_score": args.invert_score,
        "positive_indices": positive_indices,
        "prediction_splits": requested_splits,
        "selected_policy": selected,
        "valid_triage_metrics": triage_metrics(valid_pred),
        "test_triage_metrics": test_metrics,
        "baseline_test_triage_metrics": baseline_test,
        "improved_over_baseline": improved,
        "predictions": str(pred_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
