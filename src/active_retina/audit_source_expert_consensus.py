"""Patient-grouped OOF diagnostic for consensus over saved source experts.

Every candidate threshold and the expert ranking are fit on the other folds.
This remains development-only because the underlying checkpoint family and
validation predictions have already informed prior project decisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.train_quality_gated_autonomy import evaluate_policy, select_policy

SCORE_COLUMNS = ("risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin")
EXCLUDED_EYE_IDS = {"ddr_hf_specificity300:ddr_test_20170417105537128"}


def _group_key(row: dict[str, str]) -> str:
    return f"{row['source_dataset']}\0{row['patient_id']}"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            row for row in csv.DictReader(stream)
            if row.get("split") in {"valid", "validation"}
            and row.get("eye_id") not in EXCLUDED_EYE_IDS
        ]


def _stratified_group_folds(rows: list[dict[str, str]], n_folds: int) -> dict[str, int]:
    strata: dict[str, np.ndarray] = {}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    sources = sorted({row["source_dataset"] for row in rows})
    for row in rows:
        groups[_group_key(row)].append(row)
    categories = [(source, category) for source in sources for category in ("negative", "positive", "ungradable")]
    for patient, patient_rows in groups.items():
        values = np.zeros(len(categories), dtype=float)
        for row in patient_rows:
            grade = int(row["grade"])
            category = "ungradable" if grade == 5 else "positive" if grade >= 2 else "negative"
            values[categories.index((row["source_dataset"], category))] += 1
        strata[patient] = values

    totals = np.sum(list(strata.values()), axis=0)
    target = totals / n_folds
    best: tuple[float, dict[str, int]] | None = None
    for salt in range(256):
        assignment = {
            patient_group: int.from_bytes(
                hashlib.sha256(f"{salt}:{patient_group}".encode("utf-8")).digest()[:4], "big"
            ) % n_folds
            for patient_group in groups
        }
        fold_counts = np.zeros((n_folds, len(categories)), dtype=float)
        fold_sizes = np.zeros(n_folds, dtype=float)
        for patient, fold in assignment.items():
            fold_counts[fold] += strata[patient]
            fold_sizes[fold] += len(groups[patient])
        score = float(np.sum(((fold_counts - target) / np.maximum(target, 1)) ** 2))
        score += float(np.sum(((fold_sizes - len(rows) / n_folds) / max(1, len(rows) / n_folds)) ** 2))
        # Avoid calibration folds with no negatives or positives for any source.
        for index, (_, category) in enumerate(categories):
            if totals[index] >= n_folds and np.any(fold_counts[:, index] == 0):
                score += 1e6
        if best is None or score < best[0]:
            best = (score, assignment)
    if best is None:
        raise RuntimeError("Could not create patient-grouped folds")
    return best[1]


def _arrays(rows: list[dict[str, str]], score: str) -> dict[str, np.ndarray]:
    return {
        "grade": np.asarray([int(row["grade"]) for row in rows]),
        "risk": np.asarray([float(row[score]) for row in rows]),
        "quality": np.asarray([float(row["quality"]) for row in rows]),
        "quality_known": np.asarray([int(row["quality_known"]) for row in rows]),
        "ungradable": np.asarray([int(row["ungradable"]) for row in rows]),
    }


def _fit_policy(train: dict[str, np.ndarray]) -> dict[str, Any]:
    return select_policy(train)


def _action(pred: dict[str, np.ndarray], policy: dict[str, Any]) -> np.ndarray:
    return (
        (pred["risk"] >= float(policy["risk_threshold"]))
        | (pred["quality"] >= float(policy["quality_threshold"]))
    )


def audit(
    prediction_root: Path,
    n_folds: int = 5,
    top_ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> dict[str, Any]:
    paths = sorted(prediction_root.rglob("validation_predictions.csv"))
    if not paths:
        raise ValueError(f"No validation_predictions.csv files under {prediction_root}")
    candidate_rows: dict[str, dict[str, dict[str, str]]] = {}
    for path in paths:
        rows = _read(path)
        if not rows:
            continue
        mapped = {row["eye_id"]: row for row in rows}
        if len(mapped) != len(rows):
            continue
        name = f"{path.parent.name}"
        candidate_rows[name] = mapped
    if not candidate_rows:
        raise ValueError("No usable validation prediction files")

    # Anchor to the broadest all-source cohort; exclude runs evaluated on only a
    # subset, rather than shrinking the analysis to whichever source overlaps.
    reference_name = max(
        candidate_rows,
        key=lambda name: (len(candidate_rows[name]), len({r["source_dataset"] for r in candidate_rows[name].values()})),
    )
    reference = candidate_rows[reference_name]
    shared_ids = set(reference)
    candidate_rows = {
        name: mapped for name, mapped in candidate_rows.items()
        if set(mapped) == shared_ids
        and all(
            mapped[eye][key] == reference[eye][key]
            for eye in shared_ids
            for key in ("patient_id", "source_dataset", "grade", "quality_known", "ungradable")
        )
    }
    if not candidate_rows:
        raise ValueError("No candidate file matches the broadest all-source validation cohort")
    rows_by_source: dict[str, list[str]] = defaultdict(list)
    for eye_id in sorted(shared_ids):
        rows_by_source[reference[eye_id]["source_dataset"]].append(eye_id)
    shared_rows = [reference[eye_id] for source in sorted(rows_by_source) for eye_id in rows_by_source[source]]
    fold_by_patient = _stratified_group_folds(shared_rows, n_folds)

    modes = ("permissive_clear_consensus", "conservative_any_referral")
    per_k = {
        (k, mode): {"cleared_negative": 0, "negative": 0, "missed_positive": 0, "positive": 0,
                    "missed_ungradable": 0, "ungradable": 0, "cases": 0, "source_metrics": {},
                    "missed_positive_cases": []}
        for k in top_ks for mode in modes
    }
    fold_reports = []
    for fold in range(n_folds):
        train_ids = [eye for eye in shared_ids if fold_by_patient[_group_key(reference[eye])] != fold]
        test_ids = [eye for eye in shared_ids if fold_by_patient[_group_key(reference[eye])] == fold]
        test_actions: dict[tuple[int, str], dict[str, np.ndarray]] = {
            (k, mode): {} for k in top_ks for mode in modes
        }
        fold_candidates = []
        for source in sorted(rows_by_source):
            source_train_ids = [eye for eye in train_ids if reference[eye]["source_dataset"] == source]
            source_test_ids = [eye for eye in test_ids if reference[eye]["source_dataset"] == source]
            ranked = []
            for name, mapped in candidate_rows.items():
                if not all(eye in mapped for eye in source_train_ids + source_test_ids):
                    continue
                for score in SCORE_COLUMNS:
                    if any(mapped[eye].get(score) in (None, "") for eye in source_train_ids + source_test_ids):
                        continue
                    train = _arrays([mapped[eye] for eye in source_train_ids], score)
                    try:
                        policy = _fit_policy(train)
                    except ValueError:
                        continue
                    metrics = evaluate_policy(train, policy)
                    test_pred = _arrays([mapped[eye] for eye in source_test_ids], score)
                    action = _action(test_pred, policy)
                    ranked.append((
                        -float(metrics["specificity_gradable_nonreferable"]),
                        float(metrics["referral_rate"]), name, score, policy, action,
                    ))
            ranked.sort(key=lambda item: item[:4])
            if not ranked:
                raise ValueError(f"No trainable candidates for source {source}, fold {fold}")
            for k in top_ks:
                chosen = ranked[:min(k, len(ranked))]
                votes = [item[5] for item in chosen]
                # The permissive rule clears if any expert clears; the conservative rule
                # refers if any expert refers. Both are reported to expose the safety tradeoff.
                test_actions[(k, "permissive_clear_consensus")][source] = np.logical_and.reduce(votes)
                test_actions[(k, "conservative_any_referral")][source] = np.logical_or.reduce(votes)
            fold_candidates.append({
                "source": source,
                "training_patients": len({_group_key(reference[e]) for e in source_train_ids}),
                "test_patients": len({_group_key(reference[e]) for e in source_test_ids}),
                "top_candidates": [
                    {"model": item[2], "score": item[3], "calibration_specificity": -item[0],
                     "thresholds": {"risk": item[4]["risk_threshold"], "quality": item[4]["quality_threshold"]}}
                    for item in ranked[:min(max(top_ks), len(ranked))]
                ],
            })
        fold_summary = {"fold": fold, "candidates_by_source": fold_candidates, "metrics_by_k": {}}
        for k in top_ks:
            for mode in modes:
                fold_key = f"{k}:{mode}"
                fold_summary["metrics_by_k"][fold_key] = {}
                for source, ids in rows_by_source.items():
                    chosen_ids = [eye for eye in test_ids if reference[eye]["source_dataset"] == source]
                    source_rows = [reference[eye] for eye in chosen_ids]
                    action = test_actions[(k, mode)][source]
                    grades = np.asarray([int(row["grade"]) for row in source_rows])
                    qknown = np.asarray([int(row["quality_known"]) for row in source_rows]).astype(bool)
                    ungradable = np.asarray([int(row["ungradable"]) for row in source_rows]).astype(bool) & qknown
                    neg = grades <= 1
                    pos = (grades >= 2) & (grades <= 4)
                    m = {
                        "cleared_negative": int((~action[neg]).sum()), "negative": int(neg.sum()),
                        "missed_positive": int((~action[pos]).sum()), "positive": int(pos.sum()),
                        "missed_ungradable": int((~action[ungradable]).sum()), "ungradable": int(ungradable.sum()),
                        "cases": len(source_rows),
                    }
                    totals = per_k[(k, mode)]
                    for field, value in m.items():
                        totals[field] += value
                    totals["missed_positive_cases"].extend(
                        {
                            "fold": fold,
                            "eye_id": chosen_ids[index],
                            "patient_id": source_rows[index]["patient_id"],
                            "source_dataset": source,
                            "grade": int(source_rows[index]["grade"]),
                            "policy_mode": mode,
                            "top_k": k,
                        }
                        for index in np.flatnonzero(pos & ~action)
                    )
                    source_totals = totals["source_metrics"].setdefault(source, {key: 0 for key in m})
                    for field, value in m.items():
                        source_totals[field] += value
                    fold_summary["metrics_by_k"][fold_key][source] = m
                fold_summary["metrics_by_k"][fold_key]["all_sources"] = {
                    field: sum(item[field] for item in fold_summary["metrics_by_k"][fold_key].values())
                    for field in ("cleared_negative", "negative", "missed_positive", "positive",
                                  "missed_ungradable", "ungradable")
                }
        fold_reports.append(fold_summary)

    results = {}
    for (k, mode), values in per_k.items():
        negatives, positives = values["negative"], values["positive"]
        results[f"{k}:{mode}"] = {
            **values,
            "specificity": values["cleared_negative"] / negatives if negatives else None,
            "case_action_coverage": 1.0,
            "miss_rate": values["missed_positive"] / positives if positives else None,
        }
    return {
        "status": "DEVELOPMENT_ONLY_PATIENT_GROUPED_OOF_EXPERT_CONSENSUS",
        "candidate_count": len(candidate_rows),
        "common_validation_eyes": len(shared_ids),
        "fold_count": n_folds,
        "top_k_consensus_results": results,
        "folds": fold_reports,
        "claim_boundary": (
            "This is exploratory OOF analysis over repeatedly inspected validation predictions and a previously searched checkpoint family. "
            "It is not untouched external validation, clinical certification, or proof of population-wide zero risk."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default="results")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    output = Path(args.summary_output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    report = audit(Path(args.prediction_root), args.folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "common_validation_eyes": report["common_validation_eyes"],
                      "results": report["top_k_consensus_results"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
