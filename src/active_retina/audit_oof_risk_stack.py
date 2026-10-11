"""Nested patient-grouped OOF audit of a regularized stack over saved predictions.

Base model predictions are development-only and the validation cohort is exposed;
this diagnostic estimates whether combining complementary score channels is worth
testing on a future untouched cohort. It is not a certification result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from active_retina.audit_source_expert_consensus import (
    EXCLUDED_EYE_IDS,
    SCORE_COLUMNS,
    _group_key,
    _read,
    _stratified_group_folds,
)


def _load(prediction_root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, dict[str, str]]]]:
    candidates: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted(prediction_root.rglob("validation_predictions.csv")):
        rows = _read(path)
        mapped = {row["eye_id"]: row for row in rows}
        if len(mapped) == len(rows) and rows:
            candidates[path.parent.name] = mapped
    if not candidates:
        raise ValueError(f"No candidate validation predictions under {prediction_root}")
    reference_name = max(
        candidates,
        key=lambda name: (len(candidates[name]), len({r["source_dataset"] for r in candidates[name].values()})),
    )
    reference = candidates[reference_name]
    shared_ids = set(reference)
    candidates = {
        name: mapped for name, mapped in candidates.items()
        if set(mapped) == shared_ids and all(
            mapped[eye][key] == reference[eye][key]
            for eye in shared_ids
            for key in ("patient_id", "source_dataset", "grade", "quality_known", "ungradable")
        )
    }
    rows = [reference[eye] for eye in sorted(shared_ids)]
    return rows, candidates


def _source_feature_matrix(
    ids: list[str], candidates: dict[str, dict[str, dict[str, str]]]
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for candidate, mapped in sorted(candidates.items()):
        for score in (*SCORE_COLUMNS, "quality"):
            values = [mapped[eye].get(score, "") for eye in ids]
            if any(value in (None, "") for value in values):
                continue
            array = np.asarray(values, dtype=np.float64)
            if np.isfinite(array).all() and float(np.std(array)) > 1e-10:
                columns.append(array)
                names.append(f"{candidate}::{score}")
    if not columns:
        raise ValueError("No finite, nonconstant prediction features")
    return np.column_stack(columns), names


def _policy_threshold(scores: np.ndarray, grades: np.ndarray) -> float:
    positive = (grades >= 2) & (grades <= 4)
    if not positive.any():
        raise ValueError("Policy fitting requires positive referable examples")
    return float(np.min(scores[positive]))


def _fit_stack(
    x_train: np.ndarray, grades_train: np.ndarray, x_eval: np.ndarray, k: int, c: float
) -> tuple[np.ndarray, float, list[int]]:
    gradable = grades_train <= 4
    x_fit = x_train[gradable]
    y_fit = (grades_train[gradable] >= 2).astype(int)
    if len(np.unique(y_fit)) != 2:
        raise ValueError("Stack fitting needs both negative and positive labels")
    # Select only on the fitting partition using the zero-observed-miss operating point.
    ranking: list[tuple[float, float, int]] = []
    for index in range(x_fit.shape[1]):
        threshold = _policy_threshold(x_fit[:, index], grades_train[gradable])
        referred = x_fit[:, index] >= threshold
        specificity = float((~referred[y_fit == 0]).mean()) if np.any(y_fit == 0) else 0.0
        ranking.append((-specificity, float(referred.mean()), index))
    ranking.sort()
    selected = [item[2] for item in ranking[: min(k, len(ranking))]]
    scaler = StandardScaler().fit(x_fit[:, selected])
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=0)
    model.fit(scaler.transform(x_fit[:, selected]), y_fit)
    train_score = model.decision_function(scaler.transform(x_fit[:, selected]))
    threshold = _policy_threshold(train_score, grades_train[gradable])
    eval_score = model.decision_function(scaler.transform(x_eval[:, selected]))
    return eval_score, threshold, selected


def _fold_metrics(rows: list[dict[str, str]], actions: np.ndarray) -> dict[str, int | float]:
    grades = np.asarray([int(row["grade"]) for row in rows])
    negative = grades <= 1
    positive = (grades >= 2) & (grades <= 4)
    qknown = np.asarray([int(row["quality_known"]) for row in rows]).astype(bool)
    ungradable = (grades == 5) & qknown
    actions = actions.copy()
    actions[ungradable] = True
    return {
        "cleared_negative": int((~actions[negative]).sum()),
        "negative": int(negative.sum()),
        "missed_positive": int((~actions[positive]).sum()),
        "positive": int(positive.sum()),
        "missed_ungradable": int((~actions[ungradable]).sum()),
        "ungradable": int(ungradable.sum()),
    }


def audit(prediction_root: Path, folds: int = 5) -> dict[str, Any]:
    rows, candidates = _load(prediction_root)
    sources = sorted({row["source_dataset"] for row in rows})
    totals = {source: {k: 0 for k in ("cleared_negative", "negative", "missed_positive", "positive", "missed_ungradable", "ungradable")} for source in sources}
    fold_reports: list[dict[str, Any]] = []
    ks, cs = (1, 2, 4, 8, 16, 32), (0.001, 0.01, 0.1, 1.0)

    # Nested patient-grouped CV chooses stack size and regularization without using
    # the outer-fold labels. Each outer patient is predicted exactly once.
    for source in sources:
        source_rows = [row for row in rows if row["source_dataset"] == source]
        source_candidates = {
            name: {row["eye_id"]: mapped[row["eye_id"]] for row in source_rows}
            for name, mapped in candidates.items()
        }
        ids = [row["eye_id"] for row in source_rows]
        x, feature_names = _source_feature_matrix(ids, source_candidates)
        grades = np.asarray([int(row["grade"]) for row in source_rows])
        fold_map = _stratified_group_folds(source_rows, folds)
        for outer in range(folds):
            train_idx = np.asarray([i for i, row in enumerate(source_rows) if fold_map[_group_key(row)] != outer])
            eval_idx = np.asarray([i for i, row in enumerate(source_rows) if fold_map[_group_key(row)] == outer])
            train_rows = [source_rows[i] for i in train_idx]
            inner_map = _stratified_group_folds(train_rows, min(3, max(2, folds - 1)))
            settings: list[tuple[float, float, int, float]] = []
            for k in ks:
                for c in cs:
                    fold_cleared = fold_negative = fold_missed = fold_positive = 0
                    for inner in sorted(set(inner_map.values())):
                        itrain = np.asarray([i for i, row in enumerate(train_rows) if inner_map[_group_key(row)] != inner])
                        ivalid = np.asarray([i for i, row in enumerate(train_rows) if inner_map[_group_key(row)] == inner])
                        inner_train_global = train_idx[itrain]
                        inner_valid_global = train_idx[ivalid]
                        score, threshold, _ = _fit_stack(
                            x[inner_train_global], grades[inner_train_global], x[inner_valid_global], k, c
                        )
                        inner_actions = score >= threshold
                        metrics = _fold_metrics([source_rows[i] for i in inner_valid_global], inner_actions)
                        fold_cleared += int(metrics["cleared_negative"])
                        fold_negative += int(metrics["negative"])
                        fold_missed += int(metrics["missed_positive"])
                        fold_positive += int(metrics["positive"])
                    spec = fold_cleared / fold_negative if fold_negative else 0.0
                    miss_rate = fold_missed / fold_positive if fold_positive else 1.0
                    settings.append((miss_rate, -spec, k, c))
            settings.sort()
            chosen_k, chosen_c = settings[0][2], settings[0][3]
            eval_score, threshold, selected = _fit_stack(
                x[train_idx], grades[train_idx], x[eval_idx], chosen_k, chosen_c
            )
            metrics = _fold_metrics([source_rows[i] for i in eval_idx], eval_score >= threshold)
            for key in totals[source]:
                totals[source][key] += int(metrics[key])
            fold_reports.append({
                "source": source,
                "fold": outer,
                "train_patients": len({_group_key(row) for row in train_rows}),
                "eval_patients": len({_group_key(source_rows[i]) for i in eval_idx}),
                "chosen_k": chosen_k,
                "chosen_c": chosen_c,
                "threshold": threshold,
                "selected_features": [feature_names[i] for i in selected],
                "oof_scores": [
                    {
                        "eye_id": source_rows[index]["eye_id"],
                        "patient_id": source_rows[index]["patient_id"],
                        "grade": int(source_rows[index]["grade"]),
                        "quality_known": int(source_rows[index]["quality_known"]),
                        "ungradable": int(source_rows[index]["ungradable"]),
                        "score": float(eval_score[position]),
                    }
                    for position, index in enumerate(eval_idx)
                ],
                "metrics": metrics,
            })

    for source, metrics in totals.items():
        metrics["specificity"] = metrics["cleared_negative"] / metrics["negative"] if metrics["negative"] else None
        metrics["sensitivity"] = 1 - metrics["missed_positive"] / metrics["positive"] if metrics["positive"] else None
        metrics["case_action_coverage"] = 1.0
    aggregate = {key: sum(int(item[key]) for item in totals.values()) for key in (
        "cleared_negative", "negative", "missed_positive", "positive", "missed_ungradable", "ungradable"
    )}
    aggregate["specificity"] = aggregate["cleared_negative"] / aggregate["negative"] if aggregate["negative"] else None
    aggregate["sensitivity"] = 1 - aggregate["missed_positive"] / aggregate["positive"] if aggregate["positive"] else None
    aggregate["case_action_coverage"] = 1.0
    hindsight_totals = {source: {k: 0 for k in (
        "cleared_negative", "negative", "missed_positive", "positive", "missed_ungradable", "ungradable"
    )} for source in sources}
    for report in fold_reports:
        oof = report["oof_scores"]
        grades_fold = np.asarray([row["grade"] for row in oof])
        scores_fold = np.asarray([row["score"] for row in oof])
        positive_fold = (grades_fold >= 2) & (grades_fold <= 4)
        # Intentionally leaky diagnostic: use held-out positives to find the
        # threshold needed for zero misses, only to test score separability.
        if positive_fold.any():
            oracle_threshold = float(scores_fold[positive_fold].min())
            fold_source_rows = [
                {
                    "grade": str(row["grade"]),
                    "quality_known": str(row["quality_known"]),
                    "ungradable": str(row["ungradable"]),
                }
                for row in oof
            ]
            oracle = _fold_metrics(fold_source_rows, scores_fold >= oracle_threshold)
            for key in hindsight_totals[report["source"]]:
                hindsight_totals[report["source"]][key] += int(oracle[key])
    for metrics in hindsight_totals.values():
        metrics["specificity"] = metrics["cleared_negative"] / metrics["negative"] if metrics["negative"] else None
        metrics["sensitivity"] = 1 - metrics["missed_positive"] / metrics["positive"] if metrics["positive"] else None
        metrics["case_action_coverage"] = 1.0
    hindsight_aggregate = {key: sum(int(item[key]) for item in hindsight_totals.values()) for key in (
        "cleared_negative", "negative", "missed_positive", "positive", "missed_ungradable", "ungradable"
    )}
    hindsight_aggregate["specificity"] = (
        hindsight_aggregate["cleared_negative"] / hindsight_aggregate["negative"]
        if hindsight_aggregate["negative"] else None
    )
    hindsight_aggregate["sensitivity"] = (
        1 - hindsight_aggregate["missed_positive"] / hindsight_aggregate["positive"]
        if hindsight_aggregate["positive"] else None
    )
    hindsight_aggregate["case_action_coverage"] = 1.0
    return {
        "status": "DEVELOPMENT_ONLY_NESTED_PATIENT_GROUPED_OOF_STACK",
        "candidate_count": len(candidates),
        "fold_count": folds,
        "sources": totals,
        "aggregate": aggregate,
        "hindsight_zero_miss_separability_ceiling_not_a_valid_policy": {
            "sources": hindsight_totals,
            "aggregate": hindsight_aggregate,
            "explanation": "Uses each held-out fold's positive scores to choose its threshold; deliberately optimistic and invalid as an operating-point estimate.",
        },
        "folds": fold_reports,
        "claim_boundary": (
            "All base predictions come from a repeatedly inspected development validation cohort and searched model family. "
            "Nested patient grouping limits stack-selection leakage but does not create independent external validation or certification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default="results")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    output = Path(args.summary_output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    report = audit(Path(args.prediction_root), args.folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "sources": report["sources"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
