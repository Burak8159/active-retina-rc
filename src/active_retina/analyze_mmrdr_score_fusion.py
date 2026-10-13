"""Validation-only rank fusion for MMRDR cached-feature and fine-tuned scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def read_finetuned(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["image_id"] for row in rows]
    y = np.asarray([int(row["referable_dr"]) for row in rows], dtype=np.int64)
    score = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    return ids, y, score


def read_hgb(path: Path, ids: list[str]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == "hist_gradient_boost"]
    lookup = {row["image_id"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(ids):
        raise ValueError("HGB validation IDs are missing, duplicated, or misaligned")
    return np.asarray([float(lookup[image_id]["score"]) for image_id in ids], dtype=np.float64)


def measure(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    positive = score[y == 1]
    negative = score[y == 0]
    threshold = float(positive.min())
    tp = int(np.sum(positive >= threshold))
    tn = int(np.sum(negative < threshold))
    return {
        "n": int(len(y)),
        "positive": int(len(positive)),
        "negative": int(len(negative)),
        "zero_miss_threshold": threshold,
        "true_positive": tp,
        "false_negative": int(len(positive) - tp),
        "true_negative": tn,
        "false_positive": int(len(negative) - tn),
        "specificity_zero_miss": tn / len(negative),
        "sensitivity": tp / len(positive),
        "validation_auroc": float(roc_auc_score(y, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--hgb-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ids, y, ft = read_finetuned(args.finetuned)
    hgb_scores = [read_hgb(path, ids) for path in args.hgb_scores]
    with np.load("results/mmrdr_uwf_features_20261013.npz", allow_pickle=False) as cache:
        feature_splits = cache["splits"].astype(str)
        feature_ids = cache["image_ids"].astype(str).tolist()
        feature_y = cache["labels"].astype(np.int64)
    if ids != [image_id for image_id, split in zip(feature_ids, feature_splits) if split == "valid"]:
        raise ValueError("Fine-tuned score order differs from validation feature cache")
    if not np.array_equal(y, feature_y[feature_splits == "valid"]):
        raise ValueError("Fine-tuned labels differ from validation feature cache")

    ft_rank = rankdata(ft, method="average") / (len(ft) + 1.0)
    hgb_ranks = [rankdata(score, method="average") / (len(score) + 1.0) for score in hgb_scores]
    candidates = [{"candidate": "finetuned_only", **measure(y, ft_rank)}]
    candidates.extend({"candidate": f"hgb_{index + 1}_only", **measure(y, score)} for index, score in enumerate(hgb_ranks))
    for index, hgb_rank in enumerate(hgb_ranks, 1):
        for ft_weight in (0.1, 0.25, 0.5, 0.75, 0.9):
            fused = (1.0 - ft_weight) * hgb_rank + ft_weight * ft_rank
            candidates.append({"candidate": f"hgb_{index}_ft_{ft_weight:.2f}", **measure(y, fused)})
    if len(hgb_ranks) >= 2:
        for hgb2_weight in (0.25, 0.5, 0.75):
            fused = (1.0 - hgb2_weight) * hgb_ranks[0] + hgb2_weight * hgb_ranks[1]
            candidates.append({"candidate": f"hgb1_hgb2_{hgb2_weight:.2f}", **measure(y, fused)})
        candidates.append({"candidate": "all_sources_equal_rank", **measure(y, np.mean([*hgb_ranks, ft_rank], axis=0))})

    candidates.sort(key=lambda row: (float(row["specificity_zero_miss"]), float(row["validation_auroc"])), reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "MODEL_SELECTION_DIAGNOSTIC_ONLY_NOT_PATIENT_INDEPENDENT; OFFICIAL_TEST_NOT_READ",
        "split_caveat": "MMRDR internal validation is image-stratified; source patient IDs are unavailable.",
        "selection_warning": "Weights and zero-miss thresholds are assessed on the same validation partition; results are optimistic model-selection diagnostics, not certification.",
        "feature_head_files": [str(path) for path in args.hgb_scores],
        "score_transform": "within-validation empirical rank/(n+1)",
        "candidates_ranked": candidates,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
