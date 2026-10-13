"""Bounded validation-only fusion check for RETFound against the current UWF reference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def read_score_file(path: Path, model_name: str | None = None) -> tuple[dict[str, float], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if model_name is not None:
        rows = [row for row in rows if row.get("model") == model_name]
    if not rows:
        raise ValueError(f"No rows found in {path} for model={model_name}")
    score_key = "score"
    id_key = "image_id"
    label_key = "referable_dr"
    scores = {row[id_key]: float(row[score_key]) for row in rows}
    labels = {row[id_key]: int(row[label_key]) for row in rows}
    if len(scores) != len(rows) or len(labels) != len(rows):
        raise ValueError(f"Duplicate image IDs in {path} for model={model_name}")
    return scores, labels


def align(inputs: list[tuple[dict[str, float], dict[str, int]]]) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    ids = sorted(inputs[0][0])
    if not ids:
        raise ValueError("Empty validation score set")
    labels_by_id = inputs[0][1]
    for scores, labels in inputs[1:]:
        if set(scores) != set(ids) or labels != labels_by_id:
            raise ValueError("Validation score IDs/labels do not align exactly")
    y = np.asarray([labels_by_id[image_id] for image_id in ids], dtype=np.int64)
    arrays = [np.asarray([scores[image_id] for image_id in ids], dtype=np.float64) for scores, _ in inputs]
    return ids, y, arrays


def zero_miss(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    positives, negatives = score[y == 1], score[y == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("Both classes are required")
    threshold = float(positives.min())
    tn = int(np.sum(negatives < threshold))
    return {
        "n": int(len(y)),
        "positive": int(len(positives)),
        "negative": int(len(negatives)),
        "false_negative": int(np.sum(positives < threshold)),
        "true_negative": tn,
        "specificity_zero_miss": tn / len(negatives),
        "sensitivity_zero_miss": 1.0,
        "threshold": threshold,
        "validation_auroc": float(roc_auc_score(y, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uwf-hgb-scores", type=Path, required=True)
    parser.add_argument("--finetuned-scores", type=Path, required=True)
    parser.add_argument("--retfound-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hgb = read_score_file(args.uwf_hgb_scores, "hist_gradient_boost")
    finetuned = read_score_file(args.finetuned_scores)
    ids, y, raw = align([hgb, finetuned])
    hgb_rank, ft_rank = [rankdata(values, method="average") / (len(ids) + 1.0) for values in raw]
    reference = 0.75 * hgb_rank + 0.25 * ft_rank
    candidates = [{"candidate": "reference_hgb075_ft025", **zero_miss(y, reference)}]

    retfound_names = ("logreg_scaled_c0.01", "rbf_svc_l2_c10")
    retfound_inputs = [read_score_file(args.retfound_scores, name) for name in retfound_names]
    _retfound_ids, _retfound_y, retfound_raw = align([hgb, *retfound_inputs])
    for name, values in zip(retfound_names, retfound_raw[1:]):
        retfound_rank = rankdata(values, method="average") / (len(ids) + 1.0)
        for weight in (0.10, 0.25, 0.50):
            fused = (1.0 - weight) * reference + weight * retfound_rank
            candidates.append({"candidate": f"reference_plus_{name}_{weight:.2f}", **zero_miss(y, fused)})
    candidates.sort(key=lambda row: (float(row["specificity_zero_miss"]), float(row["validation_auroc"])), reverse=True)
    payload = {
        "status": "SAME_VALIDATION_MODEL_SELECTION_DIAGNOSTIC_ONLY; OFFICIAL_TEST_NOT_READ",
        "warning": "Fusion weights and operating thresholds are evaluated on the same internal image-level UWF validation split; scores are exploratory and optimistic, not certification.",
        "reference": "0.75 rank(HGB on frozen ResNet+DINO features) + 0.25 rank(448px layer-4 fine-tuned ResNet score)",
        "retfound_candidates": list(retfound_names),
        "specificity_zero_miss_target": "grade>=2 referable; all 572 validation positives must remain referred",
        "candidates_ranked": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
