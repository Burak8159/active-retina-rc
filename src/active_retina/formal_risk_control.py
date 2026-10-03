"""Finite-sample risk-control audit for ActiveRetina-RC.

This module does not use MFIDDR and does not inspect the exposed historical test
split. It selects stop/acquire/defer policies using DeepDRiD calibration
patients only, attaches exact one-sided binomial risk upper bounds, and evaluates
the selected policies once on validation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

from scipy.stats import beta

from .evaluate_active_retina_rc_policy import pair_predictions, read_predictions
from .write_fast_paper_pack import (
    BRANCHES,
    candidate_grid,
    confidence_triage_metrics,
    summarize_decisions,
    two_stage_metrics,
    write_csv,
)


ALPHAS = (0.05, 0.10, 0.20)
RISK_TARGETS = (0.05, 0.10, 0.15)
MIN_SPECIFICITY = 0.50


def binomial_upper(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound for a binomial rate."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def miss_counts(metrics: dict[str, Any]) -> tuple[int, int]:
    decided_ref = int(round(metrics["n_decided_positive_patients"]))
    risk = metrics["patient_missed_risk_decided_positive"]
    if not math.isfinite(risk):
        return 0, 0
    misses = int(round(risk * decided_ref))
    return misses, decided_ref


def add_bounds(row: dict[str, Any], metrics: dict[str, Any], n_tests: int) -> dict[str, Any]:
    misses, decided_ref = miss_counts(metrics)
    out = {**row, "cal_misses": misses, "cal_decided_positive_patients": decided_ref}
    for alpha in ALPHAS:
        out[f"risk_ucb_alpha_{alpha}"] = binomial_upper(misses, decided_ref, alpha)
        out[f"risk_ucb_bonf_alpha_{alpha}"] = binomial_upper(misses, decided_ref, alpha / max(1, n_tests))
    return out


def threshold_decisions(examples: list[dict[str, Any]], score_key: str, threshold: float, images: int) -> dict[str, Any]:
    rows = []
    for ex in examples:
        pred = 1 if ex[score_key] >= threshold else 0
        rows.append({**ex, "prediction": pred, "score": ex[score_key], "images": images, "deferred": 0, "action": score_key})
    return summarize_decisions(rows)


def enumerate_candidates(calibration: list[dict[str, Any]], max_points: int = 13) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    first_scores = candidate_grid([x["first_score"] for x in calibration], max_points)
    two_scores = candidate_grid([x["two_view_score"] for x in calibration], max_points)

    for threshold in first_scores:
        metrics = threshold_decisions(calibration, "first_score", threshold, 1)
        candidates.append(
            {
                "method": "fixed_one",
                "params": {"threshold": threshold},
                "calibration": metrics,
                "eval_fn": lambda ex, t=threshold: threshold_decisions(ex, "first_score", t, 1),
            }
        )

    for threshold in two_scores:
        metrics = threshold_decisions(calibration, "two_view_score", threshold, 2)
        candidates.append(
            {
                "method": "fixed_two",
                "params": {"threshold": threshold},
                "calibration": metrics,
                "eval_fn": lambda ex, t=threshold: threshold_decisions(ex, "two_view_score", t, 2),
            }
        )

    for low in first_scores:
        for high in first_scores:
            if low >= high:
                continue
            metrics = confidence_triage_metrics(calibration, low, high)
            candidates.append(
                {
                    "method": "confidence_triage",
                    "params": {"low": low, "high": high},
                    "calibration": metrics,
                    "eval_fn": lambda ex, lo=low, hi=high: confidence_triage_metrics(ex, lo, hi),
                }
            )

    # Two-stage policy: first view can stop, second view can resolve or defer.
    for first_low in first_scores:
        for first_high in first_scores:
            if first_low >= first_high:
                continue
            for second_low in two_scores:
                for second_high in two_scores:
                    if second_low >= second_high:
                        continue
                    metrics = two_stage_metrics(calibration, first_low, first_high, second_low, second_high)
                    candidates.append(
                        {
                            "method": "two_stage_acquire",
                            "params": {
                                "first_low": first_low,
                                "first_high": first_high,
                                "second_low": second_low,
                                "second_high": second_high,
                            },
                            "calibration": metrics,
                            "eval_fn": lambda ex, fl=first_low, fh=first_high, sl=second_low, sh=second_high: two_stage_metrics(
                                ex, fl, fh, sl, sh
                            ),
                        }
                    )

    return candidates


def compact_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_images": metrics["mean_images"],
        f"{prefix}_defer": metrics["defer_rate"],
        f"{prefix}_coverage": 1.0 - metrics["defer_rate"],
        f"{prefix}_specificity": metrics["specificity_decided"],
        f"{prefix}_miss_risk": metrics["missed_referable_risk_decided"],
        f"{prefix}_patient_miss_risk": metrics["patient_missed_risk_decided_positive"],
        f"{prefix}_positive_patients": metrics["n_positive_patients"],
        f"{prefix}_decided_positive_patients": metrics["n_decided_positive_patients"],
    }


def choose_policy(
    branch: str,
    candidates: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    alpha: float,
    target: float,
    corrected: bool,
) -> dict[str, Any]:
    n_tests = len(candidates) if corrected else 1
    feasible = []
    for item in candidates:
        cal = item["calibration"]
        misses, decided_ref = miss_counts(cal)
        ucb = binomial_upper(misses, decided_ref, alpha / n_tests)
        if (
            math.isfinite(ucb)
            and ucb <= target
            and math.isfinite(cal["specificity_decided"])
            and cal["specificity_decided"] >= MIN_SPECIFICITY
        ):
            feasible.append((item, ucb))
    if not feasible:
        return {
            "branch": branch,
            "alpha": alpha,
            "target": target,
            "correction": "bonferroni" if corrected else "single_policy",
            "status": "NO_CERTIFIED_POLICY",
            "n_candidates": len(candidates),
        }
    item, ucb = min(
        feasible,
        key=lambda z: (
            z[0]["calibration"]["mean_images"],
            z[0]["calibration"]["defer_rate"],
            -z[0]["calibration"]["specificity_decided"],
        ),
    )
    val = item["eval_fn"](validation)
    return {
        "branch": branch,
        "alpha": alpha,
        "target": target,
        "correction": "bonferroni" if corrected else "single_policy",
        "status": "CERTIFIED_ON_CALIBRATION",
        "method": item["method"],
        "params": json.dumps(item["params"], sort_keys=True),
        "calibration_patient_risk_ucb": ucb,
        "n_candidates": len(candidates),
        **compact_metrics("cal", item["calibration"]),
        **compact_metrics("val", val),
    }


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{key: row.get(key, "") for key in keys} for row in rows]


def main() -> None:
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "FORMAL_RISK_CONTROL_COMPLETED_WITHOUT_MFIDDR",
        "risk_unit": "patient-level missed referable DR among automatically decided positive-patient cases",
        "method": "exact one-sided Clopper-Pearson upper bounds on calibration patients; validation is reported once for selected policies",
        "limits": [
            "This is a finite-sample calibration audit, not external validation.",
            "Bonferroni correction is conservative because the policy grid is large.",
            "The exposed historical test split remains unused.",
            "DeepDRiD suffix mapping remains handled by two sensitivity branches.",
        ],
        "branches": [],
    }
    for branch in BRANCHES:
        rows = read_predictions(branch["predictions"])
        pairs = list(pair_predictions(rows, branch["first_view"], branch["second_view"]).values())
        calibration = [x for x in pairs if x["split"] == "calibration"]
        validation = [x for x in pairs if x["split"] == "validation"]
        candidates = enumerate_candidates(calibration)
        summary["branches"].append(
            {
                "branch": branch["name"],
                "n_calibration": len(calibration),
                "n_validation": len(validation),
                "n_candidates": len(candidates),
            }
        )
        for item in candidates:
            row = {
                "branch": branch["name"],
                "method": item["method"],
                "params": json.dumps(item["params"], sort_keys=True),
                "cal_images": item["calibration"]["mean_images"],
                "cal_defer": item["calibration"]["defer_rate"],
                "cal_specificity": item["calibration"]["specificity_decided"],
                "cal_patient_miss_risk": item["calibration"]["patient_missed_risk_decided_positive"],
            }
            audit_rows.append(add_bounds(row, item["calibration"], len(candidates)))
        for alpha in ALPHAS:
            for target in RISK_TARGETS:
                selected_rows.append(choose_policy(branch["name"], candidates, validation, alpha, target, corrected=False))
                selected_rows.append(choose_policy(branch["name"], candidates, validation, alpha, target, corrected=True))

    Path("results/formal_risk_control_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(Path("results/formal_risk_control_selected.csv"), normalize(selected_rows))
    write_csv(Path("results/formal_risk_control_candidates.csv"), normalize(audit_rows))


if __name__ == "__main__":
    main()
