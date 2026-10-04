"""Evaluate a timm model under the three-output autonomy contract.

Target:
    Increase diagnostic decision rate while preserving
    * zero false-negative nonreferable clearances
    * >=90% specificity among diagnostic decisions
    * autonomous output for every image

This supports timm model names and Hugging Face timm-hub names such as
``hf_hub:bitfount/RETFound_DR_IDRID``.
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
    counts: dict[str, int] = {}
    grade_counts: dict[tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split = row["source_split"]
            grade = int(row["grade"])
            if max_rows_per_split > 0 and counts.get(split, 0) >= max_rows_per_split:
                continue
            key = (split, grade)
            if max_rows_per_grade_per_split > 0 and grade_counts.get(key, 0) >= max_rows_per_grade_per_split:
                continue
            counts[split] = counts.get(split, 0) + 1
            grade_counts[key] = grade_counts.get(key, 0) + 1
            row["grade"] = grade
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, root: Path, grayscale_triplicate: bool) -> None:
        self.rows = rows
        self.transform = transform
        self.root = root
        self.grayscale_triplicate = grayscale_triplicate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = resolve_image_path(str(row["image_path"]), self.root)
        with Image.open(path) as img:
            if self.grayscale_triplicate:
                image = img.convert("L").convert("RGB")
            else:
                image = img.convert("RGB")
        return self.transform(image), row["image_id"], int(row["grade"]), int(row["referable_dr"]), str(path)


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_model(model_name: str, image_size: int, pretrained: bool):
    import timm

    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if image_size > 0:
        kwargs["img_size"] = image_size
    try:
        model = timm.create_model(model_name, **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        model = timm.create_model(model_name, **kwargs)
    cfg = timm.data.resolve_model_data_config(model)
    if image_size > 0:
        cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, cfg


def predict_split(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    transform: Any,
    root: Path,
    batch_size: int,
    device: torch.device,
    positive_indices: list[int],
    use_amp: bool,
    grayscale_triplicate: bool,
    invert_score: bool,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        FundusDataset(rows, transform, root, grayscale_triplicate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    out: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, image_ids, grades, ys, paths in loader:
            with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
                logits = model(images.to(device))
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 1 or logits.shape[1] == 1:
                    probs = torch.sigmoid(logits.reshape(-1, 1))
                    scores = probs[:, 0]
                else:
                    probs = torch.softmax(logits, dim=1)
                    safe_indices = [idx for idx in positive_indices if idx < probs.shape[1]]
                    if not safe_indices:
                        safe_indices = [probs.shape[1] - 1]
                    scores = probs[:, safe_indices].sum(dim=1)
            for image_id, grade, y, score, path in zip(image_ids, grades.numpy(), ys.numpy(), scores.detach().cpu().float().numpy(), paths):
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
    out = []
    for row in rows:
        rr = dict(row)
        rr["action"] = action(float(row["score"]), low, high, bool(row["quality_ok"]))
        rr["low_threshold"] = low
        rr["high_threshold"] = high
        out.append(rr)
    return out


def select_policy(valid: list[dict[str, Any]], min_specificity: float) -> dict[str, Any]:
    scores = sorted({float(r["score"]) for r in valid})
    if not scores:
        return {"status": "NO_SCORES"}
    grid = sorted(set([scores[0] - 1e-9, scores[-1] + 1e-9, *np.quantile(scores, np.linspace(0, 1, 121)).tolist()]))
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for low in grid:
        for high in grid:
            if low >= high:
                continue
            pred = apply_policy(valid, float(low), float(high))
            m = triage_metrics(pred)
            if m["false_negative_clearances"] != 0:
                continue
            if m["specificity_among_diagnostic_decisions"] < min_specificity:
                continue
            key = (
                float(m["diagnostic_decision_rate"]),
                float(m["negative_clearance_rate"]),
                float(m["referable_autonomous_referral_rate"]),
            )
            row = {"low": float(low), "high": float(high), **m}
            if best is None or key > best[0]:
                best = (key, row)
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
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--positive-indices", default="2,3,4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--grayscale-triplicate", action="store_true")
    parser.add_argument("--invert-score", action="store_true")
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--baseline-summary", default="results/three_output_autonomous_triage/summary.json")
    parser.add_argument("--max-rows-per-split", type=int, default=0)
    parser.add_argument("--max-rows-per-grade-per-split", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    rows = read_manifest(Path(args.manifest), args.max_rows_per_split, args.max_rows_per_grade_per_split)
    by_split = split_rows(rows)
    device = device_for(args.device)
    model, transform, data_config = build_model(args.model_name, args.image_size, not args.no_pretrained)
    model.to(device)
    positive_indices = [int(x) for x in args.positive_indices.split(",") if x.strip()]
    valid = predict_split(
        model,
        by_split["valid"],
        transform,
        root,
        args.batch_size,
        device,
        positive_indices,
        args.use_amp,
        args.grayscale_triplicate,
        args.invert_score,
    )
    selected = select_policy(valid, args.min_specificity)
    low = float(selected.get("low", 0.0))
    high = float(selected.get("high", 1.0))
    valid_pred = apply_policy(valid, low, high)
    test = predict_split(
        model,
        by_split["test"],
        transform,
        root,
        args.batch_size,
        device,
        positive_indices,
        args.use_amp,
        args.grayscale_triplicate,
        args.invert_score,
    )
    test_pred = apply_policy(test, low, high)
    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid_pred)
    write_predictions(pred_path, "test", test_pred)
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
        "model_name": args.model_name,
        "data_config": data_config,
        "grayscale_triplicate": args.grayscale_triplicate,
        "invert_score": args.invert_score,
        "positive_indices": positive_indices,
        "selected_policy": selected,
        "valid_triage_metrics": triage_metrics(valid_pred),
        "test_triage_metrics": test_metrics,
        "baseline_test_triage_metrics": baseline_test,
        "improved_over_baseline": improved,
        "predictions": str(pred_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
