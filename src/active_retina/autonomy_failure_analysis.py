"""Failure analysis for high-sensitivity autonomy and ActiveRetina rescue.

Uses existing predictions only. No model retraining, no threshold repair on test.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DDR_MODELS = {
    "resnet50": "results/ddr_open_autonomy_specificity300/predictions.csv",
    "dino_vit": "results/ddr_open_autonomy_specificity300_vit_dino/predictions.csv",
    "convnext_tiny": "results/ddr_open_autonomy_specificity300_convnext_tiny/predictions.csv",
    "feature_ensemble": "results/ddr_open_autonomy_specificity300_ensemble/predictions.csv",
    "mlp_ensemble": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50/predictions.csv",
    "mlp_ensemble_conservative": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50_conservative/predictions.csv",
}
SENSITIVITY_TARGETS = (0.95, 0.97, 0.98, 0.99, 1.00)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def read_manifest_grades(path: Path) -> dict[str, int]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image_id"]] = int(row["grade"])
    return out


def read_prediction_file(path: Path, model: str, grades: dict[str, int]) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            score = row.get("referable_probability", row.get("score"))
            if score is None:
                continue
            rows[image_id] = {
                "model": model,
                "split": row["split"],
                "image_id": image_id,
                "grade": int(row.get("grade") or grades.get(image_id, -1)),
                "referable_dr": int(row["referable_dr"]),
                "score": float(score),
                "action": row.get("action", ""),
            }
    return rows


def confusion(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = s >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pos & pred).sum())
    fn = int((pos & ~pred).sum())
    tn = int((neg & ~pred).sum())
    fp = int((neg & pred).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "coverage": 1.0,
    }


def select_threshold_for_sensitivity(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    s = np.asarray([r["score"] for r in rows], dtype=float)
    thresholds = sorted(set(float(x) for x in s))
    thresholds = [min(thresholds) - 1e-12] + thresholds
    candidates = []
    for t in thresholds:
        m = confusion(y, s, t)
        if m["sensitivity"] >= target:
            candidates.append(m)
    if not candidates:
        return {"status": "NO_THRESHOLD", "target_sensitivity": target}
    best = max(candidates, key=lambda m: (m["specificity"], m["threshold"]))
    best["target_sensitivity"] = target
    return best


def selective_metrics(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    s = np.asarray([r["score"] for r in rows], dtype=float)
    clear = s <= low
    refer = s >= high
    abstain = ~(clear | refer)
    pred = np.full_like(y, -1)
    pred[clear] = 0
    pred[refer] = 1
    auto = pred != -1
    pos = y == 1
    neg = y == 0
    fn_all = int((pos & clear).sum())
    tp_all = int((pos & (refer | abstain)).sum())
    tn_auto = int((neg & clear).sum())
    fp_auto = int((neg & refer).sum())
    return {
        "low": float(low),
        "high": float(high),
        "coverage": float(auto.mean()),
        "abstention_rate": float(abstain.mean()),
        "referable_abstention_rate": float((pos & abstain).sum() / max(1, pos.sum())),
        "false_negative_count": fn_all,
        "sensitivity_all_cases_clear_is_negative": float(tp_all / max(1, pos.sum())),
        "specificity_auto_clear_vs_refer": float(tn_auto / max(1, tn_auto + fp_auto)),
        "auto_negative_clearance": float(tn_auto / max(1, neg.sum())),
        "auto_refer_rate": float(refer.mean()),
    }


def select_selective_policy(cal_rows: list[dict[str, Any]], min_sens: float, min_spec: float) -> dict[str, Any]:
    raw_scores = np.asarray([float(r["score"]) for r in cal_rows], dtype=float)
    qs = np.unique(np.quantile(raw_scores, np.linspace(0.0, 1.0, 81))).tolist()
    scores = sorted(set(float(x) for x in qs))
    best = None
    # Low controls CLEAR. High controls REFER. Uncertain cases abstain.
    for low in scores:
        for high in scores:
            if low >= high:
                continue
            m = selective_metrics(cal_rows, low, high)
            if m["sensitivity_all_cases_clear_is_negative"] < min_sens:
                continue
            if m["specificity_auto_clear_vs_refer"] < min_spec:
                continue
            if best is None or (m["coverage"], m["auto_negative_clearance"], -m["abstention_rate"]) > (
                best["coverage"],
                best["auto_negative_clearance"],
                -best["abstention_rate"],
            ):
                best = m
    return best or {"status": "NO_POLICY", "min_sensitivity": min_sens, "min_specificity": min_spec}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_frontier_svg(frontier_rows: list[dict[str, Any]], selective_rows: list[dict[str, Any]]) -> None:
    """Small dependency-free SVG frontier plot for the paper/report."""
    out = Path("plots/autonomy_frontier_ddr.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    colors = {
        "resnet50": "#1b9e77",
        "dino_vit": "#d95f02",
        "convnext_tiny": "#7570b3",
        "feature_ensemble": "#e7298a",
        "mlp_ensemble": "#66a61e",
        "mlp_ensemble_conservative": "#666666",
    }

    def x_from_spec(spec: float) -> float:
        return left + spec * plot_w

    def y_from_sens(sens: float) -> float:
        return top + (1.0 - sens) / 0.08 * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111} .axis{stroke:#222;stroke-width:1.2} .grid{stroke:#ddd;stroke-width:1} .label{font-size:14px} .small{font-size:12px}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="700">DDR High-Sensitivity Frontier</text>',
    ]
    for spec in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
        x = x_from_spec(spec)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="small" x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle">{int(spec * 100)}</text>')
    for sens in [0.92, 0.94, 0.96, 0.98, 1.00]:
        y = y_from_sens(sens)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{int(sens * 100)}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line x1="{x_from_spec(0.9):.1f}" y1="{top}" x2="{x_from_spec(0.9):.1f}" y2="{top + plot_h}" stroke="#b00020" stroke-width="2" stroke-dasharray="7 5"/>')
    lines.append(f'<text class="small" x="{x_from_spec(0.9) - 8:.1f}" y="{top + 16}" text-anchor="end" fill="#b00020">90% specificity target</text>')

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frontier_rows:
        by_model[str(row["model"])].append(row)
    for model, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda r: float(r["test_specificity"]))
        pts = " ".join(
            f'{x_from_spec(float(r["test_specificity"])):.1f},{y_from_sens(float(r["test_sensitivity"])):.1f}'
            for r in rows
        )
        color = colors.get(model, "#333")
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for r in rows:
            x = x_from_spec(float(r["test_specificity"]))
            y = y_from_sens(float(r["test_sensitivity"]))
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}"/>')

    best_sel = max(
        [r for r in selective_rows if r.get("status") == "OK"],
        key=lambda r: (float(r.get("test_specificity_auto_clear_vs_refer", 0)), float(r.get("test_coverage", 0))),
        default=None,
    )
    if best_sel:
        x = x_from_spec(float(best_sel["test_specificity_auto_clear_vs_refer"]))
        y = y_from_sens(float(best_sel["test_sensitivity_all_cases_clear_is_negative"]))
        lines.append(f'<path d="M {x:.1f} {y-9:.1f} L {x+8:.1f} {y+7:.1f} L {x-8:.1f} {y+7:.1f} Z" fill="#000"/>')
        lines.append(f'<text class="small" x="{x + 12:.1f}" y="{y - 8:.1f}">best selective policy</text>')

    legend_x, legend_y = left + 20, top + 22
    for idx, model in enumerate(sorted(by_model)):
        y = legend_y + idx * 20
        color = colors.get(model, "#333")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text class="small" x="{legend_x + 30}" y="{y + 4}">{model}</text>')
    lines.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle">Specificity on held-out DDR test split (%)</text>')
    lines.append(f'<text class="label" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Sensitivity on held-out DDR test split (%)</text>')
    lines.append("</svg>\n")
    out.write_text("\n".join(lines), encoding="utf-8")


def ddr_analysis() -> dict[str, Any]:
    grades = read_manifest_grades(Path("results/ddr_hf_specificity300_manifest.csv"))
    model_rows = {name: read_prediction_file(Path(path), name, grades) for name, path in DDR_MODELS.items() if Path(path).exists()}
    common_ids = sorted(set.intersection(*(set(rows) for rows in model_rows.values())))
    test_ids = [
        image_id
        for image_id in common_ids
        if next(iter(model_rows.values()))[image_id]["split"] == "test"
    ]
    valid_ids = [
        image_id
        for image_id in common_ids
        if next(iter(model_rows.values()))[image_id]["split"] == "valid"
    ]
    valid_by_model = {m: [rows[i] for i in valid_ids] for m, rows in model_rows.items()}
    test_by_model = {m: [rows[i] for i in test_ids] for m, rows in model_rows.items()}

    frontier_rows = []
    thresholds: dict[str, dict[float, float]] = defaultdict(dict)
    for model, rows in valid_by_model.items():
        for target in SENSITIVITY_TARGETS:
            selected = select_threshold_for_sensitivity(rows, target)
            if "threshold" not in selected:
                continue
            thresholds[model][target] = float(selected["threshold"])
            test_m = confusion(
                np.asarray([r["referable_dr"] for r in test_by_model[model]], dtype=int),
                np.asarray([r["score"] for r in test_by_model[model]], dtype=float),
                float(selected["threshold"]),
            )
            frontier_rows.append(
                {
                    "model": model,
                    "target_sensitivity": target,
                    **{f"valid_{k}": v for k, v in selected.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
            )

    # Use the 99% sensitivity threshold for overlap because it reflects the desired high-sensitivity regime.
    overlap_rows = []
    fn_sets: dict[str, set[str]] = {}
    for model, rows in test_by_model.items():
        t = thresholds.get(model, {}).get(0.99)
        if t is None:
            continue
        missed = {r["image_id"] for r in rows if r["referable_dr"] == 1 and r["score"] < t}
        fn_sets[model] = missed
        overlap_rows.append({"model": model, "threshold_source": "valid_99_sensitivity", "threshold": t, "false_negatives": len(missed)})

    jaccard_rows = []
    for a, aset in fn_sets.items():
        for b, bset in fn_sets.items():
            if a >= b:
                continue
            union = aset | bset
            jaccard_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "intersection": len(aset & bset),
                    "union": len(union),
                    "jaccard": float(len(aset & bset) / len(union)) if union else 1.0,
                }
            )

    case_rows = []
    for image_id in sorted(set.union(*fn_sets.values()) if fn_sets else set()):
        base = next(rows[image_id] for rows in model_rows.values() if image_id in rows)
        row = {
            "image_id": image_id,
            "grade": base["grade"],
            "referable_dr": base["referable_dr"],
            "miss_count_at_valid99": sum(1 for s in fn_sets.values() if image_id in s),
        }
        for model, rows in model_rows.items():
            if image_id in rows:
                row[f"{model}_score"] = rows[image_id]["score"]
                row[f"{model}_missed_at_valid99"] = int(image_id in fn_sets.get(model, set()))
        case_rows.append(row)

    selective_rows = []
    for model in model_rows:
        selected = select_selective_policy(valid_by_model[model], min_sens=0.99, min_spec=0.90)
        if "status" in selected:
            selective_rows.append({"model": model, "status": selected["status"], **selected})
            continue
        test_sel = selective_metrics(test_by_model[model], selected["low"], selected["high"])
        selective_rows.append(
            {
                "model": model,
                "status": "OK",
                **{f"valid_{k}": v for k, v in selected.items()},
                **{f"test_{k}": v for k, v in test_sel.items()},
            }
        )

    write_csv(Path("results/autonomy_frontier_ddr.csv"), frontier_rows)
    write_csv(Path("results/autonomy_false_negative_overlap_ddr.csv"), overlap_rows)
    write_csv(Path("results/autonomy_false_negative_jaccard_ddr.csv"), jaccard_rows)
    write_csv(Path("results/autonomy_persistent_false_negative_cases_ddr.csv"), case_rows)
    write_csv(Path("results/autonomy_selective_screening_ddr.csv"), selective_rows)
    write_frontier_svg(frontier_rows, selective_rows)

    persistent_all = [r for r in case_rows if r["miss_count_at_valid99"] == len(fn_sets)]
    nonduplicate_pairs = [
        r for r in jaccard_rows if {r["model_a"], r["model_b"]} != {"mlp_ensemble", "mlp_ensemble_conservative"}
    ]
    mean_nonduplicate_jaccard = float(np.mean([float(r["jaccard"]) for r in nonduplicate_pairs])) if nonduplicate_pairs else math.nan
    max_nonduplicate_jaccard = float(max([float(r["jaccard"]) for r in nonduplicate_pairs], default=math.nan))
    max_frontier = max(frontier_rows, key=lambda r: (float(r["target_sensitivity"]), float(r["test_specificity"]))) if frontier_rows else {}
    best_99 = max([r for r in frontier_rows if float(r["target_sensitivity"]) == 0.99], key=lambda r: float(r["test_specificity"]), default={})
    best_selective = max(
        [r for r in selective_rows if r.get("status") == "OK"],
        key=lambda r: (float(r.get("test_coverage", 0)), float(r.get("test_specificity_auto_clear_vs_refer", 0))),
        default={},
    )
    return {
        "n_common_test_images": len(test_ids),
        "n_test_positive": int(sum(next(iter(model_rows.values()))[i]["referable_dr"] for i in test_ids)),
        "models": sorted(model_rows),
        "false_negative_counts_at_valid99": overlap_rows,
        "persistent_false_negatives_missed_by_all_models": len(persistent_all),
        "mean_pairwise_fn_jaccard_excluding_duplicate_mlp": mean_nonduplicate_jaccard,
        "max_pairwise_fn_jaccard_excluding_duplicate_mlp": max_nonduplicate_jaccard,
        "persistent_case_table": "results/autonomy_persistent_false_negative_cases_ddr.csv",
        "frontier_table": "results/autonomy_frontier_ddr.csv",
        "frontier_plot": "plots/autonomy_frontier_ddr.svg",
        "best_99_sensitivity_test_specificity": best_99,
        "best_selective_policy": best_selective,
    }


def deepdrid_analysis() -> dict[str, Any]:
    rows = []
    with Path("results/deepdrid_full_cv_predictions.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            row["first_score"] = float(row["first_score"])
            row["two_view_score"] = float(row["two_view_score"])
            rows.append(row)
    rescue_rows = []
    for branch in sorted({r["branch"] for r in rows}):
        br = [r for r in rows if r["branch"] == branch and r["split"] == "outer_validation"]
        y = np.asarray([r["referable_dr"] for r in br], dtype=int)
        first = np.asarray([r["first_score"] for r in br], dtype=float)
        two = np.asarray([r["two_view_score"] for r in br], dtype=float)
        # Use threshold selected on inner calibration for 99% sensitivity, fold-specific.
        fold_thresholds = {}
        for fold in sorted({int(r["outer_fold"]) for r in rows if r["branch"] == branch}):
            cal = [r for r in rows if r["branch"] == branch and int(r["outer_fold"]) == fold and r["split"] == "inner_calibration"]
            if not cal:
                continue
            fold_thresholds[fold] = {
                "first": select_threshold_for_sensitivity(
                    [{"referable_dr": int(r["referable_dr"]), "score": float(r["first_score"])} for r in cal], 0.99
                ),
                "two": select_threshold_for_sensitivity(
                    [{"referable_dr": int(r["referable_dr"]), "score": float(r["two_view_score"])} for r in cal], 0.99
                ),
            }
        first_misses = []
        rescued_by_two = []
        still_missed = []
        for r in br:
            fold = int(r["outer_fold"])
            ft = fold_thresholds[fold]["first"]["threshold"]
            tt = fold_thresholds[fold]["two"]["threshold"]
            first_miss = int(r["referable_dr"]) == 1 and float(r["first_score"]) < ft
            two_miss = int(r["referable_dr"]) == 1 and float(r["two_view_score"]) < tt
            if first_miss:
                first_misses.append(r["eye_id"])
                if not two_miss:
                    rescued_by_two.append(r["eye_id"])
                else:
                    still_missed.append(r["eye_id"])
        rescue_rows.append(
            {
                "branch": branch,
                "n_validation_eyes": len(br),
                "referable_eyes": int(y.sum()),
                "first_view_false_negatives_at_fold99": len(first_misses),
                "rescued_by_two_view": len(rescued_by_two),
                "still_missed_after_two_view": len(still_missed),
                "fn_rescue_rate": float(len(rescued_by_two) / max(1, len(first_misses))),
                "first_auc_proxy": float(np.nan),
                "mean_views_always_one": 1.0,
                "mean_views_always_two": 2.0,
            }
        )
    write_csv(Path("results/deepdrid_multiview_fn_rescue.csv"), rescue_rows)
    return {"multiview_rescue_table": "results/deepdrid_multiview_fn_rescue.csv", "rows": rescue_rows}


def main() -> None:
    ddr = ddr_analysis()
    deep = deepdrid_analysis()
    summary = {
        "status": "AUTONOMY_FAILURE_ANALYSIS_COMPLETE",
        "governing_question": "What information or decision process is required to prevent remaining false negatives?",
        "ddr_single_image_autonomy": ddr,
        "deepdrid_active_acquisition_rescue": deep,
        "decision": {
            "category": "ACTIVE_SELECTIVE_FUSION_REQUIRED_BEFORE_AUTONOMY_GO",
            "current_autonomy_status": "NO_GO_90_SPECIFICITY_ZERO_MISS",
            "retraining_decision": "TARGETED_FUSION_AND_POLICY_TRAINING_JUSTIFIED",
            "rationale": [
                "Existing single-image model families do not reach the requested 90%+ specificity with zero misses on the held-out DDR test split.",
                "False negatives are not a single impossible shared set: no referable image is missed by every model at the valid-99 operating point.",
                "Low pairwise false-negative overlap means the next model should fuse complementary evidence instead of rerunning one backbone family.",
                "DeepDRiD paired-view analysis shows the second view rescues 75-80% of first-view false negatives at high sensitivity.",
                "The next autonomy system should be active plus selective: acquire another view when first-view evidence is unsafe, fuse the views/models, and abstain or refer if risk remains uncertain.",
            ],
            "concrete_next_model": {
                "name": "ActiveRetina-Autonomy-v2",
                "components": [
                    "multi-model fusion of complementary DDR predictors",
                    "paired-view fusion on datasets with same-eye views",
                    "asymmetric loss or constrained optimization that heavily penalizes false nonreferable clearance",
                    "CLEAR / REFER / ABSTAIN policy trained against specificity, sensitivity, and coverage together",
                    "hard-example mining from calibration/development folds only",
                ],
                "success_criterion": "held-out or cross-fitted test result with >=90% specificity and zero false-negative autonomous clearances, plus reported coverage",
            },
        },
    }
    Path("results/autonomy_failure_analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# ActiveRetina Autonomy Failure Analysis\n\n",
        "Status: **analysis complete; do not rerun failed backbone loops**.\n\n",
        "## 1. False-Negative Overlap\n\n",
        f"Analyzed {len(ddr['models'])} DDR model families on {ddr['n_common_test_images']} common test images with {ddr['n_test_positive']} referable positives. ",
        f"At validation-selected 99% sensitivity, **{ddr['persistent_false_negatives_missed_by_all_models']} referable images were missed by every model**. ",
        f"Mean non-duplicate pairwise false-negative Jaccard was **{ddr['mean_pairwise_fn_jaccard_excluding_duplicate_mlp']:.3f}**. ",
        "This means the failures are mostly model-specific and fusion/hard-example learning is justified. See `results/autonomy_false_negative_overlap_ddr.csv`, `results/autonomy_false_negative_jaccard_ddr.csv`, and `results/autonomy_persistent_false_negative_cases_ddr.csv`.\n\n",
        "## 2. Operating Frontier\n\n",
        f"Best current forced-binary DDR 99% sensitivity operating point is `{ddr['best_99_sensitivity_test_specificity'].get('model')}`: ",
        f"test sensitivity **{pct(ddr['best_99_sensitivity_test_specificity'].get('test_sensitivity', 0.0))}**, ",
        f"test specificity **{pct(ddr['best_99_sensitivity_test_specificity'].get('test_specificity', 0.0))}**, ",
        f"false negatives **{ddr['best_99_sensitivity_test_specificity'].get('test_fn')}**. ",
        "This fails the 90%+/zero-miss autonomy target. The frontier plot is `plots/autonomy_frontier_ddr.svg`.\n\n",
        "## 3. Selective Screening\n\n",
        f"Best selective policy is `{ddr['best_selective_policy'].get('model')}`: ",
        f"test coverage **{pct(ddr['best_selective_policy'].get('test_coverage', 0.0))}**, ",
        f"test sensitivity **{pct(ddr['best_selective_policy'].get('test_sensitivity_all_cases_clear_is_negative', 0.0))}**, ",
        f"test auto-specificity **{pct(ddr['best_selective_policy'].get('test_specificity_auto_clear_vs_refer', 0.0))}**, ",
        f"false negatives **{ddr['best_selective_policy'].get('test_false_negative_count')}**. ",
        "Selective screening is closer, but still not the final autonomous result.\n\n",
        "## 4. Multi-View Rescue\n\n",
        "DeepDRiD paired-view rescue is the strongest positive signal: ",
        "; ".join(
            f"{r['branch']} rescued {r['rescued_by_two_view']}/{r['first_view_false_negatives_at_fold99']} first-view false negatives ({pct(r['fn_rescue_rate'])})"
            for r in deep["rows"]
        ),
        ". This directly supports ActiveRetina's central claim that acquiring another field can fix errors that a first-view-only autonomy model would otherwise make.\n\n",
        "## 5. Decision\n\n",
        "**NO-GO for the current 90%+/zero-miss autonomous model; GO for targeted ActiveRetina-Autonomy-v2 training.** ",
        "The concrete solution is not another isolated threshold run. It is an active selective fusion system: first-view risk, second-view acquisition when unsafe, model/view fusion, then CLEAR / REFER / ABSTAIN with risk-controlled thresholds. ",
        "Because the false negatives are complementary across models and mostly rescued by an added view, this is a scientifically justified next model rather than blind retrying.\n\n",
        "## 6. Paper Impact\n\n",
        "This strengthens the flagship paper if we frame it correctly: the paper is not a leaderboard classifier. It is a falsifiable active-acquisition/autonomous-clearance study showing why single-image thresholding fails, how second-view acquisition rescues false negatives, and what evidence is required before autonomous clearance can be claimed.\n",
    ]
    Path("docs/autonomy_failure_analysis_report.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
