"""Nested patient-fold test of a source-specific expert-score selector.

All source cohorts are previously inspected development data. This estimates
whether retrospective score complementarity is learnable; it is not a clinical
certificate or independent validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from active_retina.audit_expert_score_oracle_union import (
    ROOT,
    EXCLUDED_COLUMNS,
    _current_recipe_scores,
    _source_rows,
    read_csv,
)


SOURCES = ("DDR", "DeepDRiD", "DRTiD")
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
UNION_SHRINKS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def all_aligned_candidates(results: Path, eye_ids: list[str], base: np.ndarray):
    candidates: list[tuple[str, np.ndarray]] = [("current_fused_recipe", base)]
    seen = {base.tobytes()}
    for path in sorted(results.rglob("*.csv")):
        try:
            rows = read_csv(path)
        except (OSError, UnicodeError, csv.Error):
            continue
        if not rows or "eye_id" not in rows[0]:
            continue
        by_id = {row["eye_id"]: row for row in rows}
        if len(by_id) != len(rows) or not set(eye_ids).issubset(by_id):
            continue
        for column in rows[0]:
            if column in EXCLUDED_COLUMNS:
                continue
            try:
                values = np.asarray([float(by_id[i][column]) for i in eye_ids], dtype=float)
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(values).all() or np.ptp(values) <= 1e-12:
                continue
            key = values.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((f"{path.relative_to(results)}::{column}", values))
    return candidates


def rank_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_rank = np.empty_like(train, dtype=float)
    test_rank = np.empty_like(test, dtype=float)
    for column in range(train.shape[1]):
        values = train[:, column]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = ((start + 1) + end) / (2 * len(values))
            start = end
        train_rank[:, column] = ranks
        test_rank[:, column] = np.searchsorted(np.sort(values), test[:, column], side="right") / len(values)
    return train_rank, test_rank


def score_candidates(results: Path, source: str, eye_ids: list[str],
                     candidate_mode: str = "oracle_contributors") -> tuple[np.ndarray, list[str]]:
    candidates: list[tuple[str, np.ndarray]] = []
    base, _ = _current_recipe_scores(results, source, eye_ids)
    if candidate_mode == "all_aligned":
        # Rank transforms are fitted inside folds; this pool is assembled without labels.
        candidates = all_aligned_candidates(results, eye_ids, base)
        return np.column_stack([values for _, values in candidates]), [name for name, _ in candidates]
    if candidate_mode != "oracle_contributors":
        raise ValueError(f"Unknown candidate mode: {candidate_mode}")
    candidates.append(("current_fused_recipe", base))
    oracle = json.loads((results / "expert_score_oracle_union_20261012.json").read_text(encoding="utf-8"))
    selected = oracle["source_results"][source]["scores_that_added_at_least_one_case"]
    cache: dict[Path, dict[str, dict[str, str]]] = {}
    for item in selected:
        if item == "current_fused_recipe":
            continue
        relpath, column = item.split("::", 1)
        path = results / relpath
        if path not in cache:
            rows = read_csv(path)
            cache[path] = {row["eye_id"]: row for row in rows}
        mapping = cache[path]
        if not all(eye_id in mapping and column in mapping[eye_id] for eye_id in eye_ids):
            continue
        values = np.asarray([float(mapping[i][column]) for i in eye_ids], dtype=float)
        if np.isfinite(values).all() and np.ptp(values) > 1e-12:
            candidates.append((item, values))
    return np.column_stack([values for _, values in candidates]), [name for name, _ in candidates]


def split_folds(labels: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    assignment = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        assignment[test] = fold
    if np.any(assignment < 0):
        raise ValueError("Patient-grouped folds did not assign every example")
    return assignment


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, c: float, family: str):
    tx, vx = rank_train_test(train_x, test_x)
    if family == "logistic":
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=4000, solver="lbfgs")
        model.fit(tx, train_y)
        train_score = model.decision_function(tx)
        test_score = model.decision_function(vx)
    elif family == "shallow_tree":
        model = DecisionTreeClassifier(max_depth=int(c), min_samples_leaf=max(6, int(0.04 * len(train_y))),
                                       class_weight="balanced", random_state=20261012)
        model.fit(tx, train_y)
        train_score = model.predict_proba(tx)[:, 1]
        test_score = model.predict_proba(vx)[:, 1]
    elif family == "pairwise_rank":
        pos, neg = tx[train_y == 1], tx[train_y == 0]
        pair_differences = (pos[:, None, :] - neg[None, :, :]).reshape(-1, tx.shape[1])
        pair_x = np.concatenate([pair_differences, -pair_differences], axis=0)
        pair_y = np.concatenate([np.ones(len(pair_differences)), np.zeros(len(pair_differences))])
        model = LinearSVC(C=c, fit_intercept=False, dual="auto", max_iter=5000)
        model.fit(pair_x, pair_y)
        train_score = tx @ model.coef_.ravel()
        test_score = vx @ model.coef_.ravel()
    else:
        raise ValueError(f"Unknown selector family: {family}")
    threshold = float(train_score[train_y == 1].min())
    return test_score, threshold


def crossfit_threshold(x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float,
                       family: str, folds: int, seed: int) -> float:
    assignment = split_folds(y, groups, folds, seed)
    oof = np.full(len(y), np.nan)
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        scores, _ = fit_predict(x[train], y[train], x[test], c, family)
        oof[test] = scores
    if not np.isfinite(oof[y == 1]).all():
        raise ValueError("Cross-fitted threshold has missing positive scores")
    return float(oof[y == 1].min())


def crossfit_single_threshold(values: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              folds: int, seed: int) -> float:
    assignment = split_folds(y, groups, folds, seed)
    oof = np.full(len(y), np.nan)
    for fold in range(folds):
        val = assignment == fold
        train = ~val
        tx, vx = rank_train_test(values[train, None], values[val, None])
        oof[val] = vx[:, 0]
    if not np.isfinite(oof[y == 1]).all():
        raise ValueError("Cross-fitted expert threshold has missing positive scores")
    return float(oof[y == 1].min())


def select_single_expert(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         folds: int, seed: int) -> tuple[int, dict]:
    assignment = split_folds(y, groups, folds, seed)
    trials = []
    for column in range(x.shape[1]):
        misses = tn = negatives = 0
        for fold in range(folds):
            val = assignment == fold
            train = ~val
            threshold = crossfit_single_threshold(x[train, column], y[train], groups[train],
                                                  min(3, folds - 1), seed + 500 + fold)
            _, val_rank = rank_train_test(x[train, column, None], x[val, column, None])
            clear = val_rank[:, 0] < threshold
            misses += int(np.sum((y[val] == 1) & clear))
            tn += int(np.sum((y[val] == 0) & clear))
            negatives += int(np.sum(y[val] == 0))
        trials.append({"column": column, "misses": misses, "true_negatives": tn,
                       "negatives": negatives, "specificity": tn / max(1, negatives)})
    chosen = max(trials, key=lambda t: (-t["misses"], t["specificity"]))
    return int(chosen["column"]), {"selected": chosen, "grid": trials}


def crossfit_union_thresholds(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              folds: int, seed: int) -> np.ndarray:
    assignment = split_folds(y, groups, folds, seed)
    oof = np.full_like(x, np.nan, dtype=float)
    for fold in range(folds):
        val = assignment == fold
        train = ~val
        _, val_rank = rank_train_test(x[train], x[val])
        oof[val] = val_rank
    if not np.isfinite(oof[y == 1]).all():
        raise ValueError("Cross-fitted union calibration is missing positive scores")
    return np.min(oof[y == 1], axis=0)


def select_union_shrink(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        folds: int, seed: int) -> tuple[float, dict]:
    assignment = split_folds(y, groups, folds, seed)
    trials = []
    for shrink in UNION_SHRINKS:
        misses = tn = negatives = 0
        for fold in range(folds):
            val = assignment == fold
            train = ~val
            thresholds = crossfit_union_thresholds(x[train], y[train], groups[train],
                                                   min(3, folds - 1), seed + 700 + fold)
            _, val_rank = rank_train_test(x[train], x[val])
            clear = np.any(val_rank < (shrink * thresholds)[None, :], axis=1)
            misses += int(np.sum((y[val] == 1) & clear))
            tn += int(np.sum((y[val] == 0) & clear))
            negatives += int(np.sum(y[val] == 0))
        trials.append({"shrink": shrink, "misses": misses, "true_negatives": tn,
                       "negatives": negatives, "specificity": tn / max(1, negatives)})
    chosen = max(trials, key=lambda t: (-t["misses"], t["specificity"]))
    return float(chosen["shrink"]), {"selected": chosen, "grid": trials}


def metrics(labels: np.ndarray, forced: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, int | float]:
    clear = (scores < thresholds) & ~forced
    positive, negative = labels == 1, labels == 0
    return {
        "cases": int(len(labels)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "misses": int(np.sum(positive & clear)),
        "true_negatives": int(np.sum(negative & clear)),
        "specificity": float(np.sum(negative & clear) / max(1, negative.sum())),
        "binary_coverage": 1.0,
    }


def inner_select(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> tuple[float, dict]:
    assignment = split_folds(y, groups, folds, seed)
    trials = []
    grids = (("logistic", C_VALUES), ("pairwise_rank", C_VALUES),
             ("shallow_tree", (1, 2, 3, 4)))
    for family, c_values in grids:
        for c in c_values:
            miss = tn = negatives = 0
            for fold in range(folds):
                val = assignment == fold
                tr = ~val
                scores, _ = fit_predict(x[tr], y[tr], x[val], c, family)
                threshold = crossfit_threshold(x[tr], y[tr], groups[tr], c, family,
                                               min(3, folds - 1), 9000 + seed + fold)
                m = metrics(y[val], np.zeros(val.sum(), dtype=bool), scores, np.full(val.sum(), threshold))
                miss += m["misses"]
                tn += m["true_negatives"]
                negatives += m["negative"]
            trials.append({"family": family, "C": c, "misses": miss, "true_negatives": tn,
                           "negatives": negatives, "specificity": tn / max(1, negatives)})
    chosen = max(trials, key=lambda t: (-t["misses"], t["specificity"], -t["C"]))
    return float(chosen["C"]), str(chosen["family"]), {"selected": chosen, "grid": trials}


def evaluate_source(results: Path, source: str, cases: dict[str, dict[str, str]], folds: int,
                    seed: int, candidate_mode: str):
    eye_ids = sorted(cases)
    labels = np.asarray([int(cases[i].get("referable", cases[i].get("referable_dr"))) for i in eye_ids])
    patient_ids = np.asarray([cases[i]["patient_id"] for i in eye_ids])
    _, forced = _current_recipe_scores(results, source, eye_ids)
    x, feature_names = score_candidates(results, source, eye_ids, candidate_mode)
    candidate_paths = sorted({name.split("::", 1)[0] for name in feature_names if "::" in name})
    source_file_hashes = {
        rel: hashlib.sha256((results / rel).read_bytes()).hexdigest()
        for rel in candidate_paths
    }
    eligible = ~forced
    x_fit, y_fit, group_fit = x[eligible], labels[eligible], patient_ids[eligible]
    outer = split_folds(y_fit, group_fit, folds, seed + 1)
    out_score = np.full(len(labels), np.nan)
    out_threshold = np.full(len(labels), np.nan)
    single_score = np.full(len(labels), np.nan)
    single_threshold = np.full(len(labels), np.nan)
    union_clear = np.zeros(len(labels), dtype=bool)
    details = []
    for fold in range(folds):
        test_fit = outer == fold
        train_fit = ~test_fit
        c, family, inner = inner_select(x_fit[train_fit], y_fit[train_fit], group_fit[train_fit], min(folds - 1, 4), seed + 100 + fold)
        scores, _ = fit_predict(x_fit[train_fit], y_fit[train_fit], x_fit[test_fit], c, family)
        threshold = crossfit_threshold(x_fit[train_fit], y_fit[train_fit], group_fit[train_fit],
                                       c, family, min(4, folds - 1), seed + 1000 + fold)
        out_score[np.flatnonzero(eligible)[test_fit]] = scores
        out_threshold[np.flatnonzero(eligible)[test_fit]] = threshold
        selected_column, single_selection = select_single_expert(
            x_fit[train_fit], y_fit[train_fit], group_fit[train_fit],
            min(folds - 1, 4), seed + 2000 + fold)
        single_threshold_value = crossfit_single_threshold(
            x_fit[train_fit, selected_column], y_fit[train_fit], group_fit[train_fit],
            min(4, folds - 1), seed + 3000 + fold)
        _, single_test = rank_train_test(x_fit[train_fit, selected_column, None], x_fit[test_fit, selected_column, None])
        test_indexes = np.flatnonzero(eligible)[test_fit]
        single_score[test_indexes] = single_test[:, 0]
        single_threshold[test_indexes] = single_threshold_value
        shrink, union_selection = select_union_shrink(
            x_fit[train_fit], y_fit[train_fit], group_fit[train_fit],
            min(folds - 1, 4), seed + 4000 + fold)
        union_thresholds = crossfit_union_thresholds(
            x_fit[train_fit], y_fit[train_fit], group_fit[train_fit],
            min(4, folds - 1), seed + 5000 + fold)
        _, union_test = rank_train_test(x_fit[train_fit], x_fit[test_fit])
        union_clear[test_indexes] = np.any(union_test < (shrink * union_thresholds)[None, :], axis=1)
        union_selection["calibration_thresholds"] = dict(zip(feature_names, union_thresholds.tolist()))
        details.append({"outer_fold": fold, "selected_family": family, "selected_C": c, "train_n": int(train_fit.sum()),
                        "test_n": int(test_fit.sum()), "inner_selection": inner,
                        "selected_single_expert": feature_names[selected_column],
                        "single_expert_inner_selection": single_selection,
                        "union_inner_selection": union_selection})
    out_threshold[forced] = np.inf
    result = metrics(labels, forced, out_score, out_threshold)
    result["cases"] = int(len(labels))
    result["unique_patients"] = int(len(set(patient_ids)))
    single_threshold[forced] = np.inf
    single_result = metrics(labels, forced, single_score, single_threshold)
    single_result["cases"] = int(len(labels))
    single_result["unique_patients"] = int(len(set(patient_ids)))
    result["nested_single_expert_selector"] = single_result
    union_result = {
        "cases": int(len(labels)), "unique_patients": int(len(set(patient_ids))),
        "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
        "misses": int(np.sum((labels == 1) & union_clear & ~forced)),
        "true_negatives": int(np.sum((labels == 0) & union_clear & ~forced)),
        "specificity": float(np.sum((labels == 0) & union_clear & ~forced) / max(1, np.sum(labels == 0))),
        "binary_coverage": 1.0,
    }
    result["nested_crossfit_or_union"] = union_result
    baseline, baseline_forced = _current_recipe_scores(results, source, eye_ids)
    base_clear, threshold = _zero_miss_baseline(labels, baseline)
    base_clear &= ~baseline_forced
    result["baseline_true_negatives_same_cohort"] = int(np.sum(base_clear & (labels == 0)))
    result["baseline_specificity_same_cohort"] = float(np.mean(base_clear[labels == 0]))
    result["features"] = feature_names
    result["candidate_channel_count"] = len(feature_names) - 1
    result["candidate_panel_sha256"] = hashlib.sha256(
        "\n".join(feature_names).encode("utf-8")
    ).hexdigest()
    result["candidate_file_sha256"] = source_file_hashes
    result["folds"] = details
    return result


def _zero_miss_baseline(labels: np.ndarray, score: np.ndarray):
    threshold = float(score[labels == 1].min())
    return score < threshold, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/expert_selector_nested_crossfit_20261012.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--candidate-mode", choices=("oracle_contributors", "all_aligned"),
                        default="oracle_contributors",
                        help="Compare the historical label-screened shortlist with every complete aligned score channel.")
    args = parser.parse_args()

    source_cases = _source_rows(args.results_dir)
    report = {"status": "EXPOSED_DATA_NESTED_PATIENT_CROSSFIT_DEVELOPMENT_ONLY",
              "method": "candidate_set_contamination_sensitivity",
              "candidate_mode": args.candidate_mode,
              "claim_boundary": "All source labels and base-model outputs have been inspected in prior development. The all-aligned mode avoids label-based candidate screening in this analysis, but the artifact universe and cohort are historically exposed. Nested patient folds limit selector self-fitting; they do not provide independent or clinical confirmation.",
              "outer_folds": args.folds, "seed": args.seed, "sources": {}}
    for source in SOURCES:
        report["sources"][source] = evaluate_source(args.results_dir, source, source_cases[source], args.folds, args.seed, args.candidate_mode)
    report["pooled"] = {
        "cases": sum(v["cases"] for v in report["sources"].values()),
        "unique_patients": sum(v["unique_patients"] for v in report["sources"].values()),
        "negatives": sum(v["negative"] for v in report["sources"].values()),
        "misses": sum(v["misses"] for v in report["sources"].values()),
        "true_negatives": sum(v["true_negatives"] for v in report["sources"].values()),
        "specificity": sum(v["true_negatives"] for v in report["sources"].values()) / sum(v["negative"] for v in report["sources"].values()),
        "coverage": 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pooled": report["pooled"],
                      "sources": {k: {m: v[m] for m in ("true_negatives", "negative", "misses", "specificity", "baseline_true_negatives_same_cohort", "baseline_specificity_same_cohort")} for k, v in report["sources"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
