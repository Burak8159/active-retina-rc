"""Track-B rescue analysis for DeepDRiD two-view adaptive acquisition.

Consumes the existing full-CV prediction file and produces:
- second-view value distribution,
- mapping-sensitivity summary,
- deployable value-policy frontier trained on inner-calibration folds only.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score


SEED = 20261004


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "branch": row["branch"],
                    "outer_fold": int(row["outer_fold"]),
                    "split": row["split"],
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "first_score": float(row["first_score"]),
                    "two_view_score": float(row["two_view_score"]),
                    "first_grade_pred": int(row["first_grade_pred"]),
                    "two_view_grade_pred": int(row["two_view_grade_pred"]),
                }
            )
    return rows


def entropy_binary(p: float) -> float:
    p = min(max(float(p), 1e-8), 1.0 - 1e-8)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def value_row(row: dict[str, Any]) -> dict[str, Any]:
    y = row["referable_dr"]
    first_loss = (y - row["first_score"]) ** 2
    two_loss = (y - row["two_view_score"]) ** 2
    delta = first_loss - two_loss
    out = dict(row)
    out["brier_loss_first"] = first_loss
    out["brier_loss_two"] = two_loss
    out["delta_brier_reduction"] = delta
    out["second_view_helped"] = int(delta > 0)
    out["first_entropy"] = entropy_binary(row["first_score"])
    out["first_uncertainty"] = 1.0 - abs(row["first_score"] - 0.5) * 2.0
    out["first_grade_pred_abs_error_proxy"] = abs(row["first_grade_pred"] - 2)
    return out


def features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                r["first_score"],
                r["first_entropy"],
                r["first_uncertainty"],
                float(r["first_grade_pred"]),
                float(r["first_grade_pred"] >= 2),
            ]
            for r in rows
        ],
        dtype=np.float32,
    )


def metric_block(rows: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    s = np.asarray([r[score_key] for r in rows], dtype=float)
    pred = (s >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "n": int(len(rows)),
        "auroc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan"),
        "auprc": float(average_precision_score(y, s)) if len(set(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, s)),
        "sensitivity_at_0_5": float(tp / max(1, tp + fn)),
        "specificity_at_0_5": float(tn / max(1, tn + fp)),
        "macro_f1_at_0_5": float(f1_score(y, pred, average="macro")),
    }


def bootstrap_ci(rows: list[dict[str, Any]], score_key: str, metric: str, n: int = 1000) -> dict[str, float]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    patients = np.asarray(sorted(by_patient))
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(n):
        sample = [x for p in rng.choice(patients, size=len(patients), replace=True) for x in by_patient[str(p)]]
        try:
            val = metric_block(sample, score_key)[metric]
        except Exception:
            continue
        if math.isfinite(val):
            reps.append(val)
    lo, hi = np.quantile(reps, [0.025, 0.975]) if reps else (float("nan"), float("nan"))
    return {"observed": metric_block(rows, score_key)[metric], "ci_low": float(lo), "ci_high": float(hi)}


def train_value_policy(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fold in sorted({r["outer_fold"] for r in branch_rows}):
        cal = [r for r in branch_rows if r["outer_fold"] == fold and r["split"] == "inner_calibration"]
        val = [r for r in branch_rows if r["outer_fold"] == fold and r["split"] == "outer_validation"]
        if len(cal) < 10 or not val:
            continue
        model = GradientBoostingRegressor(random_state=SEED + fold, n_estimators=100, max_depth=2, learning_rate=0.05)
        model.fit(features(cal), np.asarray([r["delta_brier_reduction"] for r in cal], dtype=float))
        pred_delta = model.predict(features(val))
        for row, pred in zip(val, pred_delta):
            out.append({**row, "predicted_delta": float(pred)})
    return out


def apply_value_policy(rows: list[dict[str, Any]], cost: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        acquire = row["predicted_delta"] > cost
        out.append(
            {
                **row,
                "policy_score": row["two_view_score"] if acquire else row["first_score"],
                "acquired_second_view": int(acquire),
                "views": 2 if acquire else 1,
            }
        )
    return out


def threshold_policy(rows: list[dict[str, Any]], key: str, threshold: float, mode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if mode == "entropy":
            acquire = row["first_entropy"] >= threshold
        elif mode == "uncertainty":
            acquire = row["first_uncertainty"] >= threshold
        elif mode == "risk_band":
            acquire = threshold <= row["first_score"] <= 1.0 - threshold
        else:
            raise ValueError(mode)
        out.append(
            {
                **row,
                key: row["two_view_score"] if acquire else row["first_score"],
                "acquired_second_view": int(acquire),
                "views": 2 if acquire else 1,
            }
        )
    return out


def frontier_rows(rows: list[dict[str, Any]], branch: str) -> list[dict[str, Any]]:
    policies = []
    policies.append({"branch": branch, "policy": "always_stop", "cost_or_threshold": None, "mean_views": 1.0, **metric_block(rows, "first_score")})
    policies.append({"branch": branch, "policy": "always_acquire", "cost_or_threshold": None, "mean_views": 2.0, **metric_block(rows, "two_view_score")})

    value_rows = train_value_policy(rows)
    if value_rows:
        costs = np.quantile([r["predicted_delta"] for r in value_rows], np.linspace(0.05, 0.95, 19))
        for cost in sorted(set(float(x) for x in costs)):
            applied = apply_value_policy(value_rows, cost)
            policies.append(
                {
                    "branch": branch,
                    "policy": "trained_value_policy",
                    "cost_or_threshold": cost,
                    "mean_views": float(np.mean([r["views"] for r in applied])),
                    "acquire_rate": float(np.mean([r["acquired_second_view"] for r in applied])),
                    **metric_block(applied, "policy_score"),
                }
            )

    validation = [r for r in rows if r["split"] == "outer_validation"]
    for mode, grid in (
        ("entropy", np.linspace(0.05, 0.69, 17)),
        ("uncertainty", np.linspace(0.05, 0.95, 19)),
        ("risk_band", np.linspace(0.05, 0.45, 17)),
    ):
        for threshold in grid:
            applied = threshold_policy(validation, "policy_score", float(threshold), mode)
            policies.append(
                {
                    "branch": branch,
                    "policy": f"{mode}_acquire",
                    "cost_or_threshold": float(threshold),
                    "mean_views": float(np.mean([r["views"] for r in applied])),
                    "acquire_rate": float(np.mean([r["acquired_second_view"] for r in applied])),
                    **metric_block(applied, "policy_score"),
                }
            )
    return policies


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_branch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [r for r in rows if r["split"] == "outer_validation"]
    deltas = np.asarray([r["delta_brier_reduction"] for r in validation], dtype=float)
    by_grade = {}
    for grade in sorted({r["grade"] for r in validation}):
        vals = np.asarray([r["delta_brier_reduction"] for r in validation if r["grade"] == grade], dtype=float)
        by_grade[str(grade)] = {
            "n": int(len(vals)),
            "mean_delta": float(vals.mean()) if len(vals) else float("nan"),
            "fraction_helped": float((vals > 0).mean()) if len(vals) else float("nan"),
        }
    return {
        "n_validation_eyes": len(validation),
        "first_view": {m: bootstrap_ci(validation, "first_score", m) for m in ("auroc", "auprc", "brier")},
        "two_view": {m: bootstrap_ci(validation, "two_view_score", m) for m in ("auroc", "auprc", "brier")},
        "delta_brier_reduction": {
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "p10": float(np.quantile(deltas, 0.10)),
            "p90": float(np.quantile(deltas, 0.90)),
            "fraction_positive": float((deltas > 0).mean()),
            "fraction_near_zero_abs_lt_0_01": float((np.abs(deltas) < 0.01).mean()),
            "fraction_negative": float((deltas < 0).mean()),
            "by_grade": by_grade,
        },
    }


def main() -> None:
    input_path = Path("results/deepdrid_full_cv_predictions.csv")
    rows = [value_row(r) for r in read_rows(input_path)]
    value_path = Path("results/deepdrid_track_b_second_view_value.csv")
    policy_path = Path("results/deepdrid_track_b_policy_frontier.csv")
    summary_path = Path("results/deepdrid_track_b_rescue_summary.json")
    write_csv(value_path, rows)

    summary = {
        "status": "DEEPDRID_TRACK_B_RESCUE_COMPLETE",
        "backbone_role": "Frozen ResNet-50 ImageNet embeddings are frozen as Track-B diagnostic infrastructure; backbone hunting is no longer the flagship task.",
        "mapping_branches": {
            "suffix2_initial": "current visual-audit mapping: suffix_1=optic_disc, suffix_2=macula; first view is macula",
            "suffix1_initial": "swapped sensitivity mapping: suffix_1=macula, suffix_2=optic_disc; first view is macula",
        },
        "branches": {},
        "artifacts": {
            "value_rows": str(value_path),
            "policy_frontier": str(policy_path),
            "source_predictions": str(input_path),
        },
    }
    all_policy_rows = []
    for branch in sorted({r["branch"] for r in rows}):
        branch_rows = [r for r in rows if r["branch"] == branch]
        summary["branches"][branch] = summarize_branch(branch_rows)
        all_policy_rows.extend(frontier_rows(branch_rows, branch))
    write_csv(policy_path, all_policy_rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
