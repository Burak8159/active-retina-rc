"""Patient-grouped threshold sensitivity analysis for a fixed score model.

The model checkpoint was selected using the full validation split before this
analysis. Thus, held-out folds cross-fit only the threshold, not model selection.
This is a post-selection development diagnostic, not unbiased cross-validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_validation_rows(prediction_path: Path, manifest_path: Path) -> list[dict]:
    predictions = {}
    with prediction_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if image_id in predictions:
                raise ValueError(f"duplicate prediction image_id: {image_id}")
            predictions[image_id] = row

    rows = []
    seen = set()
    with manifest_path.open(newline="") as handle:
        for manifest_row in csv.DictReader(handle):
            if manifest_row["source_split"] != "valid":
                continue
            image_id = manifest_row["image_id"]
            if image_id in seen:
                raise ValueError(f"duplicate validation manifest image_id: {image_id}")
            seen.add(image_id)
            prediction = predictions.get(image_id)
            if prediction is None:
                raise ValueError(f"missing prediction for validation image_id: {image_id}")
            if not manifest_row["patient_id"]:
                raise ValueError(f"missing patient_id for validation image_id: {image_id}")
            label = int(prediction["referable_dr"])
            if label not in (0, 1):
                raise ValueError(f"referable_dr must be binary for image_id: {image_id}")
            rows.append(
                {
                    "image_id": image_id,
                    "patient_id": manifest_row["patient_id"],
                    "label": label,
                    "score": float(prediction["score"]),
                }
            )

    extra = set(predictions) - seen
    if extra:
        raise ValueError(f"predictions include non-validation rows ({len(extra)} extra)")
    if not rows:
        raise ValueError("no validation rows found")
    return rows


def grouped_folds(rows: list[dict], n_folds: int) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["patient_id"]].append(index)
    if len(groups) < n_folds:
        raise ValueError(f"need at least {n_folds} patient groups")

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for _, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(range(n_folds), key=lambda index: (sizes[index], index))
        folds[fold].extend(indices)
        sizes[fold] += len(indices)
    return folds


def confusion(rows: list[dict], indices: list[int], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for index in indices:
        row = rows[index]
        predicted_positive = row["score"] >= threshold
        if row["label"] == 1:
            counts["tp" if predicted_positive else "fn"] += 1
        else:
            counts["fp" if predicted_positive else "tn"] += 1
    return counts


def accuracy_selected_threshold(rows: list[dict], indices: list[int]) -> float:
    scores = sorted({rows[index]["score"] for index in indices})
    candidates = [scores[0] - 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 1.0)

    def correct(threshold: float) -> int:
        return sum(
            (rows[index]["score"] >= threshold) == bool(rows[index]["label"])
            for index in indices
        )

    return max(
        candidates,
        key=lambda threshold: (
            correct(threshold),
            -abs(threshold),
        ),
    )


def run(prediction_path: Path, manifest_path: Path, n_folds: int) -> dict:
    rows = load_validation_rows(prediction_path, manifest_path)
    folds = grouped_folds(rows, n_folds)
    assigned = [index for fold in folds for index in fold]
    if sorted(assigned) != list(range(len(rows))):
        raise AssertionError("fold assignment must include each row exactly once")

    out_of_fold = [None] * len(rows)
    fold_results = []
    for fold_index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = [index for index in range(len(rows)) if index not in heldout_set]
        threshold = accuracy_selected_threshold(rows, train)
        counts = confusion(rows, heldout, threshold)
        for index in heldout:
            out_of_fold[index] = int(rows[index]["score"] >= threshold)
        fold_results.append(
            {
                "fold": fold_index,
                "threshold": threshold,
                "n_train": len(train),
                "n_heldout": len(heldout),
                "heldout_counts": counts,
            }
        )

    labels = [row["label"] for row in rows]
    pooled = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for label, prediction in zip(labels, out_of_fold):
        pooled["tp" if label == 1 and prediction == 1 else
               "fn" if label == 1 else
               "fp" if prediction == 1 else "tn"] += 1
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    return {
        "status": "POST_SELECTION_PATIENT_GROUPED_THRESHOLD_CROSSFIT_DIAGNOSTIC",
        "model": "frozen ResNet-50 candidate; no model refit in this analysis",
        "checkpoint_selection": "epoch selected using the full validation split before threshold cross-fitting",
        "threshold_objective": "maximize classification accuracy on threshold-training folds",
        "grouping": "patient_id from validation manifest",
        "n_folds": n_folds,
        "n_images": len(rows),
        "n_patients": len({row["patient_id"] for row in rows}),
        "positive_images": n_positive,
        "negative_images": n_negative,
        "folds": fold_results,
        "pooled_out_of_fold": pooled,
        "accuracy": (pooled["tp"] + pooled["tn"]) / len(rows),
        "sensitivity": pooled["tp"] / n_positive if n_positive else None,
        "specificity": pooled["tn"] / n_negative if n_negative else None,
        "misses": pooled["fn"],
        "coverage": 1.0,
        "test_data_used": False,
        "clinical_validation": False,
        "claim_boundary": (
            "Post-selection development diagnostic: only threshold selection is cross-fitted. "
            "The checkpoint epoch was selected using the full validation labels, so held-out "
            "folds do not remove model-selection leakage. Not an unbiased CV estimate, "
            "independent external validation, clinical validation, or a zero-miss result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = run(args.predictions, args.manifest, args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
