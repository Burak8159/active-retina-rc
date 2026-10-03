from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import read_yaml_like, write_json
from .evaluate_gates import auc_binary, average_precision, brier_score


REQUIRED_COLUMNS = {
    "patient_id",
    "eye_id",
    "view_set",
    "grade",
    "referable_dr",
    "referable_score",
    "fold_id",
    "prediction_scope",
}

ALLOWED_CROSSFIT_SCOPES = {"oof", "out_of_fold", "crossfit", "cross_fitted"}
ALLOWED_HELDOUT_SCOPES = {"heldout", "heldout_test", "sealed_test"}


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"Prediction file is missing required columns: {missing}")
        rows = []
        for row in reader:
            scope = row["prediction_scope"].strip().lower()
            if scope not in ALLOWED_CROSSFIT_SCOPES | ALLOWED_HELDOUT_SCOPES:
                raise SystemExit(
                    "DeepDRiD evaluation requires cross-fitted/out-of-fold or sealed held-out predictions; "
                    f"found prediction_scope={row['prediction_scope']!r}."
                )
            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "view_set": row["view_set"],
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "referable_score": float(row["referable_score"]),
                    "fold_id": row["fold_id"],
                    "prediction_scope": row["prediction_scope"],
                }
            )
    return rows


def _metrics(values: dict[str, tuple[int, float]]) -> dict[str, float]:
    keys = sorted(values)
    y_true = [values[k][0] for k in keys]
    score = [values[k][1] for k in keys]
    return {
        "n_eyes": len(keys),
        "auroc": auc_binary(y_true, score),
        "auprc": average_precision(y_true, score),
        "brier": brier_score(y_true, score),
    }


def _bootstrap(values: dict[str, tuple[int, float]], n: int, seed: int, metric: str) -> dict[str, float]:
    observed = _metrics(values)[metric]
    keys = sorted(values)
    rng = random.Random(seed)
    reps = []
    for _ in range(n):
        sampled = [rng.choice(keys) for _ in keys]
        sample_values = {f"{i}_{key}": values[key] for i, key in enumerate(sampled)}
        val = _metrics(sample_values)[metric]
        if not math.isnan(val):
            reps.append(val)
    reps.sort()
    if not reps:
        return {"observed": observed, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "observed": observed,
        "ci_low": reps[int(0.025 * (len(reps) - 1))],
        "ci_high": reps[int(0.975 * (len(reps) - 1))],
    }


def _loss(label: int, score: float) -> float:
    return (float(label) - float(score)) ** 2


def _paired_rows(rows: list[dict[str, Any]], first_view: str, second_view: str) -> dict[str, dict[str, Any]]:
    first_key = first_view
    both_key = "+".join(sorted((first_view, second_view)))
    by_eye: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_eye[row["eye_id"]][row["view_set"]] = row
    paired = {}
    for eye_id, pred_by_set in by_eye.items():
        if first_key in pred_by_set and both_key in pred_by_set:
            first = pred_by_set[first_key]
            both = pred_by_set[both_key]
            if first["patient_id"] != both["patient_id"] or first["referable_dr"] != both["referable_dr"]:
                raise SystemExit(f"Inconsistent paired predictions for eye_id={eye_id}")
            paired[eye_id] = {"first": first, "both": both}
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    first_view = cfg["evaluation"]["fixed_first_view"]
    second_view = cfg["evaluation"]["second_view"]
    pred_path = Path(args.predictions or cfg["evaluation"]["prediction_input"])
    output = Path(args.output or cfg["evaluation"]["output"])

    if not pred_path.exists():
        write_json(
            output,
            {
                "status": "BLOCKED_CROSSFIT_PREDICTIONS_MISSING",
                "dataset": cfg["dataset"]["name"],
                "track": cfg["project"]["track"],
                "view_mapping": cfg["dataset"].get("view_mapping", {}),
                "view_mapping_status": cfg["dataset"].get("view_mapping_status", "UNKNOWN"),
                "required_prediction_file": str(pred_path),
                "required_prediction_columns": sorted(REQUIRED_COLUMNS),
                "required_view_sets": [first_view, "+".join(sorted((first_view, second_view)))],
                "reason": "Track B second-view value and stopping analysis require out-of-fold/cross-fitted predictions. No empirical metrics were fabricated because the required prediction file is not present.",
            },
        )
        return

    rows = _read_predictions(pred_path)
    non_crossfit_rows = [
        row for row in rows if row["prediction_scope"].strip().lower() not in ALLOWED_CROSSFIT_SCOPES
    ]
    paired = _paired_rows(rows, first_view, second_view)
    if not paired:
        raise SystemExit("No paired first-view and two-view predictions found.")

    first_values = {eye: (p["first"]["referable_dr"], p["first"]["referable_score"]) for eye, p in paired.items()}
    both_values = {eye: (p["both"]["referable_dr"], p["both"]["referable_score"]) for eye, p in paired.items()}
    hindsight_stop_values = {}
    second_view_value = []
    for eye, pair in paired.items():
        label = pair["first"]["referable_dr"]
        first_score = pair["first"]["referable_score"]
        both_score = pair["both"]["referable_score"]
        first_loss = _loss(label, first_score)
        both_loss = _loss(label, both_score)
        use_second = both_loss < first_loss
        hindsight_stop_values[eye] = (label, both_score if use_second else first_score)
        second_view_value.append(
            {
                "eye_id": eye,
                "patient_id": pair["first"]["patient_id"],
                "referable_dr": label,
                "first_view_score": first_score,
                "two_view_score": both_score,
                "brier_loss_reduction_from_second_view": first_loss - both_loss,
                "hindsight_second_view_helped": use_second,
                "fold_id": pair["first"]["fold_id"],
            }
        )

    n_boot = int(cfg["evaluation"]["bootstrap"]["n"])
    seed = int(cfg["evaluation"]["bootstrap"]["seed"])
    write_json(
        output,
        {
            "status": "ACTIVE_RETINA_RC_PAIRED_VIEW_VALUE_EVALUATED",
            "dataset": cfg["dataset"]["name"],
            "track": cfg["project"]["track"],
            "view_mapping": cfg["dataset"].get("view_mapping", {}),
            "view_mapping_status": cfg["dataset"].get("view_mapping_status", "UNKNOWN"),
            "view_mapping_note": cfg["dataset"].get("view_mapping_note", ""),
            "fixed_first_view": first_view,
            "second_view": second_view,
            "n_paired_eyes": len(paired),
            "single_first_view": {
                **_metrics(first_values),
                "auroc_ci": _bootstrap(first_values, n_boot, seed, "auroc"),
            },
            "two_view": {
                **_metrics(both_values),
                "auroc_ci": _bootstrap(both_values, n_boot, seed, "auroc"),
            },
            "cross_fitted_hindsight_stop_oracle": {
                **_metrics(hindsight_stop_values),
                "auroc_ci": _bootstrap(hindsight_stop_values, n_boot, seed, "auroc"),
                "mean_brier_loss_reduction": mean(
                    row["brier_loss_reduction_from_second_view"] for row in second_view_value
                ),
                "fraction_eyes_second_view_helped": mean(
                    1.0 if row["hindsight_second_view_helped"] else 0.0 for row in second_view_value
                ),
            },
            "second_view_value_rows": second_view_value,
            "interpretation_limits": [
                "DeepDRiD has two views and cannot establish multi-step non-myopic planning value.",
                "The hindsight stop oracle uses true labels only for retrospective analysis of cross-fitted or sealed held-out predictions; it is not a deployable stopping policy.",
                "If visual and swapped suffix mappings produce materially different conclusions, field mapping remains a blocker pending official confirmation.",
            ],
            "prediction_scope_note": {
                "n_non_crossfit_rows": len(non_crossfit_rows),
                "allowed_non_crossfit_use": "sealed held-out predictions may be used for final test reporting, but not for training, calibration, or hindsight policy selection.",
            },
        },
    )


if __name__ == "__main__":
    main()
