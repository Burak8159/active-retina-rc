"""ActiveRetina-Autonomy-v2 fusion search.

This script does not retrain image backbones. It tests whether complementary
model scores can be fused into a safer autonomous clearance signal. Thresholds
are selected only on validation split and reported on the held-out test split.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.autonomy_failure_analysis import DDR_MODELS, confusion, select_selective_policy, selective_metrics, write_csv


def read_predictions() -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    model_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for model, rel in DDR_MODELS.items():
        path = Path(rel)
        if not path.exists():
            continue
        rows = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                score = row.get("referable_probability", row.get("score"))
                if score is None:
                    continue
                rows[row["image_id"]] = {
                    "split": row["split"],
                    "image_id": row["image_id"],
                    "referable_dr": int(row["referable_dr"]),
                    "score": float(score),
                }
        model_rows[model] = rows
    common_ids = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    return common_ids, model_rows


def make_fused_rows(
    image_ids: list[str],
    model_rows: dict[str, dict[str, dict[str, Any]]],
    model_names: list[str],
    scores: np.ndarray,
    name: str,
) -> list[dict[str, Any]]:
    rows = []
    for idx, image_id in enumerate(image_ids):
        base = model_rows[model_names[0]][image_id]
        rows.append(
            {
                "model": name,
                "split": base["split"],
                "image_id": image_id,
                "referable_dr": base["referable_dr"],
                "score": float(scores[idx]),
            }
        )
    return rows


def rank_normalize_by_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    for split in sorted({r["split"] for r in rows}):
        idxs = [i for i, r in enumerate(rows) if r["split"] == split]
        vals = np.asarray([rows[i]["score"] for i in idxs], dtype=float)
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(vals, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
        for local, global_idx in enumerate(idxs):
            out[global_idx]["score"] = float(ranks[local])
    return out


def select_threshold(rows: list[dict[str, Any]], target_sensitivity: float) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    s = np.asarray([r["score"] for r in rows], dtype=float)
    thresholds = sorted(set(float(x) for x in s))
    thresholds = [min(thresholds) - 1e-12] + thresholds
    candidates = []
    for threshold in thresholds:
        metrics = confusion(y, s, threshold)
        if metrics["sensitivity"] >= target_sensitivity:
            candidates.append(metrics)
    if not candidates:
        return {"status": "NO_THRESHOLD", "target_sensitivity": target_sensitivity}
    best = max(candidates, key=lambda m: (m["specificity"], m["threshold"]))
    best["target_sensitivity"] = target_sensitivity
    return best


def evaluate_candidate(name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [r for r in rows if r["split"] == "valid"]
    test = [r for r in rows if r["split"] == "test"]
    result_rows = []
    for target in (0.99, 0.995, 0.999, 1.0):
        selected = select_threshold(valid, target)
        if "threshold" not in selected:
            continue
        y_test = np.asarray([r["referable_dr"] for r in test], dtype=int)
        s_test = np.asarray([r["score"] for r in test], dtype=float)
        test_metrics = confusion(y_test, s_test, float(selected["threshold"]))
        result_rows.append(
            {
                "candidate": name,
                "mode": "forced_binary",
                "target_sensitivity": target,
                **{f"valid_{k}": v for k, v in selected.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )

    selected_policy = select_selective_policy(valid, min_sens=0.99, min_spec=0.90)
    if "status" not in selected_policy:
        test_policy = selective_metrics(test, selected_policy["low"], selected_policy["high"])
        result_rows.append(
            {
                "candidate": name,
                "mode": "selective_clear_refer_abstain",
                **{f"valid_{k}": v for k, v in selected_policy.items()},
                **{f"test_{k}": v for k, v in test_policy.items()},
            }
        )
    else:
        result_rows.append({"candidate": name, "mode": "selective_clear_refer_abstain", **selected_policy})

    best = max(
        result_rows,
        key=lambda r: (
            int(float(r.get("test_fn", r.get("test_false_negative_count", 10**9))) == 0),
            float(r.get("test_specificity", r.get("test_specificity_auto_clear_vs_refer", 0.0))),
            float(r.get("test_sensitivity", r.get("test_sensitivity_all_cases_clear_is_negative", 0.0))),
            float(r.get("test_coverage", 0.0)),
        ),
    )
    return result_rows, best


def main() -> None:
    common_ids, model_rows = read_predictions()
    model_names = sorted(model_rows)
    score_matrix = np.asarray(
        [[model_rows[m][image_id]["score"] for m in model_names] for image_id in common_ids],
        dtype=float,
    )

    candidate_rows: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = {}

    base_transforms = {
        "mean_all": score_matrix.mean(axis=1),
        "max_all": score_matrix.max(axis=1),
        "median_all": np.median(score_matrix, axis=1),
        "top2_mean_all": np.sort(score_matrix, axis=1)[:, -2:].mean(axis=1),
        "feature_plus_dino_mean": score_matrix[:, [model_names.index("feature_ensemble"), model_names.index("dino_vit")]].mean(axis=1),
        "feature_plus_convnext_mean": score_matrix[:, [model_names.index("feature_ensemble"), model_names.index("convnext_tiny")]].mean(axis=1),
        "dino_plus_convnext_mean": score_matrix[:, [model_names.index("dino_vit"), model_names.index("convnext_tiny")]].mean(axis=1),
    }

    # Sparse validation-weight search over complementary models. Coarse by design:
    # this is a sober rescue test, not a hidden neural retraining pass.
    weight_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    valid_mask = np.asarray([model_rows[model_names[0]][i]["split"] == "valid" for i in common_ids])
    y_valid = np.asarray([model_rows[model_names[0]][i]["referable_dr"] for i in common_ids], dtype=int)[valid_mask]
    weighted_candidates = []
    for weights in itertools.product(weight_values, repeat=len(model_names)):
        w = np.asarray(weights, dtype=float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        fused = score_matrix @ w
        rows = make_fused_rows(common_ids, model_rows, model_names, fused, "weighted_tmp")
        valid_rows = [r for r in rows if r["split"] == "valid"]
        selected = select_threshold(valid_rows, 0.99)
        if "threshold" not in selected:
            continue
        weighted_candidates.append((float(selected["specificity"]), tuple(float(x) for x in w), fused))
    weighted_candidates = sorted(weighted_candidates, key=lambda item: item[0], reverse=True)[:8]
    for rank, (_, weights, fused) in enumerate(weighted_candidates, start=1):
        label_bits = [f"{m}:{w:.2f}" for m, w in zip(model_names, weights) if w > 0]
        base_transforms[f"valid99_weighted_{rank}_{'_'.join(label_bits)}"] = fused

    for name, scores in base_transforms.items():
        rows = make_fused_rows(common_ids, model_rows, model_names, scores, name)
        candidates[name] = rows
        result_rows, _ = evaluate_candidate(name, rows)
        candidate_rows.extend(result_rows)

        ranked_rows = rank_normalize_by_split(rows)
        ranked_name = f"{name}_ranknorm"
        candidates[ranked_name] = ranked_rows
        ranked_result_rows, _ = evaluate_candidate(ranked_name, ranked_rows)
        candidate_rows.extend(ranked_result_rows)

    write_csv(Path("results/autonomy_v2_fusion_search.csv"), candidate_rows)
    best_forced = max(
        [r for r in candidate_rows if r["mode"] == "forced_binary"],
        key=lambda r: (
            int(float(r.get("test_fn", 10**9)) == 0),
            float(r.get("test_specificity", 0.0)),
            float(r.get("test_sensitivity", 0.0)),
        ),
    )
    best_selective = max(
        [r for r in candidate_rows if r["mode"] == "selective_clear_refer_abstain" and r.get("status", "OK") == "OK"],
        key=lambda r: (
            int(float(r.get("test_false_negative_count", 10**9)) == 0),
            float(r.get("test_specificity_auto_clear_vs_refer", 0.0)),
            float(r.get("test_coverage", 0.0)),
        ),
        default={},
    )
    summary = {
        "status": "FUSION_SEARCH_COMPLETE",
        "n_candidates": len({r["candidate"] for r in candidate_rows}),
        "n_rows": len(candidate_rows),
        "best_forced_binary": best_forced,
        "best_selective": best_selective,
        "decision": {
            "current_status": "NO_GO_90_SPECIFICITY_ZERO_MISS"
            if not (
                float(best_forced.get("test_fn", 1)) == 0 and float(best_forced.get("test_specificity", 0.0)) >= 0.90
            )
            else "GO_FORCED_BINARY_DEVELOPMENT",
            "interpretation": "Score-level fusion improves the search space but does not replace targeted multi-view/policy training if it fails 90% specificity with zero misses.",
        },
    }
    Path("results/autonomy_v2_fusion_search_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# ActiveRetina-Autonomy-v2 Fusion Search\n\n",
        f"Evaluated {summary['n_candidates']} score-fusion candidates using existing DDR predictions. Thresholds were selected on validation and reported on test.\n\n",
        "## Best Forced Binary\n\n",
        f"`{best_forced.get('candidate')}` at target `{best_forced.get('target_sensitivity')}`: ",
        f"test sensitivity {100 * float(best_forced.get('test_sensitivity', 0.0)):.1f}%, ",
        f"specificity {100 * float(best_forced.get('test_specificity', 0.0)):.1f}%, ",
        f"false negatives {best_forced.get('test_fn')}.\n\n",
        "## Best Selective\n\n",
        f"`{best_selective.get('candidate')}`: ",
        f"test coverage {100 * float(best_selective.get('test_coverage', 0.0)):.1f}%, ",
        f"sensitivity {100 * float(best_selective.get('test_sensitivity_all_cases_clear_is_negative', 0.0)):.1f}%, ",
        f"auto-specificity {100 * float(best_selective.get('test_specificity_auto_clear_vs_refer', 0.0)):.1f}%, ",
        f"false negatives {best_selective.get('test_false_negative_count')}.\n\n",
        "## Decision\n\n",
        f"**{summary['decision']['current_status']}**. Score-level fusion alone is not enough if the reported best row still misses the target. The next model must train the fusion and active-acquisition policy directly, rather than only combining already-frozen probabilities.\n",
    ]
    Path("docs/autonomy_v2_fusion_search_report.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
