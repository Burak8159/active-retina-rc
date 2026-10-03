"""Quantify the gap from selective autonomy to every-image autonomy.

The product target is strict: every image receives CLEAR or REFER, zero
referable cases are cleared, and specificity is at least 90%. This script
measures whether the existing frozen DDR score family contains enough
separation to support that target, including optimistic test-label ceilings
that are not deployable but are useful as diagnostics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.autonomy_failure_analysis import DDR_MODELS, write_csv


POLICY_PATH = Path("results/autonomous_mode_90_0_policy.json")


def ranknorm(values: np.ndarray, splits: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for split in sorted(set(splits.tolist())):
        idx = splits == split
        vals = values[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(vals, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
        out[idx] = ranks
    return out


def load_scores() -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model, rel in DDR_MODELS.items():
        rows = {}
        with Path(rel).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["image_id"]] = {
                    "split": row["split"],
                    "label": int(row["referable_dr"]),
                    "score": float(row.get("referable_probability", row.get("score"))),
                }
        per_model[model] = rows
    models = sorted(per_model)
    image_ids = sorted(set.intersection(*(set(rows) for rows in per_model.values())))
    splits = np.asarray([per_model[models[0]][image_id]["split"] for image_id in image_ids])
    y = np.asarray([per_model[models[0]][image_id]["label"] for image_id in image_ids], dtype=int)
    scores = np.asarray([[per_model[m][image_id]["score"] for m in models] for image_id in image_ids], dtype=float)
    return models, image_ids, splits, y, scores


def candidate_scores(scores: np.ndarray, splits: np.ndarray) -> dict[str, np.ndarray]:
    candidates = {
        "mean_all": scores.mean(axis=1),
        "median_all": np.median(scores, axis=1),
        "min_all": scores.min(axis=1),
        "max_all": scores.max(axis=1),
        "top2_mean_all": np.sort(scores, axis=1)[:, -2:].mean(axis=1),
    }
    for idx, model in enumerate(sorted(DDR_MODELS)):
        candidates[model] = scores[:, idx]
    for name, values in list(candidates.items()):
        candidates[f"{name}_ranknorm"] = ranknorm(values, splits)
    return candidates


def oracle_zero_miss_full_coverage(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pos = y == 1
    neg = y == 0
    min_positive_score = float(score[pos].min())
    clear = score < min_positive_score
    refer = ~clear
    false_negative = int((pos & clear).sum())
    true_negative = int((neg & clear).sum())
    false_positive = int((neg & refer).sum())
    true_positive = int((pos & refer).sum())
    specificity = true_negative / max(1, int(neg.sum()))
    return {
        "oracle_threshold": min_positive_score,
        "false_negative_clearances": false_negative,
        "specificity": specificity,
        "sensitivity": true_positive / max(1, int(pos.sum())),
        "true_negative_clearances": true_negative,
        "false_positive_referrals": false_positive,
        "true_positive_referrals": true_positive,
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "distance_to_90_specificity_points": max(0.0, 0.90 - specificity),
    }


def selective_policy_abstention_rows(
    image_ids: list[str],
    splits: np.ndarray,
    y: np.ndarray,
    policy_score: np.ndarray,
) -> list[dict[str, Any]]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    low = float(policy["thresholds"]["clear_if_score_lte"])
    high = float(policy["thresholds"]["refer_if_score_gte"])
    rows = []
    for image_id, split, yy, score in zip(image_ids, splits, y, policy_score):
        if split != "test":
            continue
        if score <= low:
            action = "CLEAR"
        elif score >= high:
            action = "REFER"
        else:
            action = "ABSTAIN"
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "referable_dr": int(yy),
                "policy_score": float(score),
                "action": action,
            }
        )
    return rows


def main() -> None:
    models, image_ids, splits, y, scores = load_scores()
    candidates = candidate_scores(scores, splits)
    test = splits == "test"
    rows = []
    for name, score in candidates.items():
        m = oracle_zero_miss_full_coverage(y[test], score[test])
        rows.append({"candidate": name, **m})
    rows = sorted(rows, key=lambda r: r["specificity"], reverse=True)
    write_csv(Path("results/full_autonomy_oracle_gap.csv"), rows)

    policy_name = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["candidate"]
    policy_score = candidates[policy_name]
    action_rows = selective_policy_abstention_rows(image_ids, splits, y, policy_score)
    write_csv(Path("results/selective_policy_test_actions.csv"), action_rows)
    abstain = [r for r in action_rows if r["action"] == "ABSTAIN"]
    abstain_pos = sum(int(r["referable_dr"]) for r in abstain)
    abstain_neg = len(abstain) - abstain_pos
    best = rows[0]
    summary = {
        "status": "NO_GO_EVERY_IMAGE_AUTONOMY_WITH_CURRENT_SCORE_FAMILY",
        "reason": "Even optimistic test-label oracle thresholds cannot approach 90% specificity at zero misses.",
        "best_oracle_candidate": best,
        "n_test_images": len(action_rows),
        "n_test_abstained_by_current_selective_policy": len(abstain),
        "abstained_positive_images": abstain_pos,
        "abstained_negative_images": abstain_neg,
        "current_selective_policy": str(POLICY_PATH),
        "oracle_gap_table": "results/full_autonomy_oracle_gap.csv",
        "test_action_table": "results/selective_policy_test_actions.csv",
        "required_fix": [
            "Train an image-level model specifically on hard abstained negatives and subtle referable positives.",
            "Add view/quality evidence for ambiguous images instead of forcing the current score family.",
            "Freeze a new every-image model only after it reaches zero validation misses and >=90% validation specificity, then test once.",
        ],
    }
    Path("results/full_autonomy_gap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Full Autonomy Gap Analysis\n\n",
        "Target: every test image receives CLEAR or REFER, with zero referable cases cleared and at least 90% specificity.\n\n",
        f"Status: **{summary['status']}**\n\n",
        f"Best optimistic oracle candidate: `{best['candidate']}`\n\n",
        f"Best optimistic zero-miss specificity: **{100 * float(best['specificity']):.1f}%**\n\n",
        f"Distance to 90% specificity: **{100 * float(best['distance_to_90_specificity_points']):.1f} percentage points**\n\n",
        f"Current selective policy abstains on **{len(abstain)} / {len(action_rows)}** test images.\n\n",
        f"Abstained referable images: **{abstain_pos}**; abstained nonreferable images: **{abstain_neg}**.\n\n",
        "Interpretation: the existing frozen score family does not contain enough information to safely force every case into an autonomous decision. The next valid fix is image-level hard-case training or additional view/quality evidence, not another threshold search.\n",
    ]
    Path("docs/full_autonomy_gap_analysis.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
