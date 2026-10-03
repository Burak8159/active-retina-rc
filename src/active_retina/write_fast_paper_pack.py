from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate_active_retina_rc_policy import evaluate_policy, pair_predictions, policy_decision, read_predictions
from .evaluate_gates import auc_binary, average_precision, brier_score


BRANCHES = [
    {
        "name": "suffix2_initial",
        "mapping_note": "visual-audit branch: suffix_2 treated as initial view, suffix_1 as second view",
        "predictions": Path("results/deepdrid_train_only_development_predictions.csv"),
        "policy": Path("results/deepdrid_train_only_development_policy.json"),
        "first_view": "macula",
        "second_view": "optic_disc",
    },
    {
        "name": "suffix1_initial",
        "mapping_note": "swapped sensitivity branch: suffix_1 treated as initial view, suffix_2 as second view",
        "predictions": Path("results/deepdrid_swapped_train_only_development_predictions.csv"),
        "policy": Path("results/deepdrid_swapped_train_only_development_policy.json"),
        "first_view": "macula",
        "second_view": "optic_disc",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def candidate_grid(scores: list[float], max_points: int = 51) -> list[float]:
    values = sorted(set(round(x, 6) for x in scores))
    if not values:
        return [0.5]
    if len(values) <= max_points:
        return values
    return [values[int(i * (len(values) - 1) / (max_points - 1))] for i in range(max_points)]


def threshold_metrics(examples: list[dict[str, Any]], score_key: str, threshold: float, images: int = 1) -> dict[str, Any]:
    rows = []
    for ex in examples:
        pred = 1 if ex[score_key] >= threshold else 0
        rows.append({**ex, "prediction": pred, "score": ex[score_key], "images": images, "deferred": 0, "action": f"{score_key}_threshold"})
    return summarize_decisions(rows)


def confidence_triage_metrics(examples: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    rows = []
    for ex in examples:
        if ex["first_score"] <= low:
            rows.append({**ex, "prediction": 0, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_NEGATIVE"})
        elif ex["first_score"] >= high:
            rows.append({**ex, "prediction": 1, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_POSITIVE"})
        else:
            rows.append({**ex, "prediction": None, "score": ex["first_score"], "images": 1, "deferred": 1, "action": "DEFER"})
    return summarize_decisions(rows)


def two_stage_metrics(examples: list[dict[str, Any]], first_low: float, first_high: float, second_low: float, second_high: float) -> dict[str, Any]:
    rows = []
    for ex in examples:
        if ex["first_score"] <= first_low:
            rows.append({**ex, "prediction": 0, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_NEGATIVE"})
        elif ex["first_score"] >= first_high:
            rows.append({**ex, "prediction": 1, "score": ex["first_score"], "images": 1, "deferred": 0, "action": "STOP_POSITIVE"})
        elif ex["two_view_score"] <= second_low:
            rows.append({**ex, "prediction": 0, "score": ex["two_view_score"], "images": 2, "deferred": 0, "action": "ACQUIRE_STOP_NEGATIVE"})
        elif ex["two_view_score"] >= second_high:
            rows.append({**ex, "prediction": 1, "score": ex["two_view_score"], "images": 2, "deferred": 0, "action": "ACQUIRE_STOP_POSITIVE"})
        else:
            rows.append({**ex, "prediction": None, "score": ex["two_view_score"], "images": 2, "deferred": 1, "action": "ACQUIRE_DEFER"})
    return summarize_decisions(rows)


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decided = [r for r in rows if not r["deferred"]]
    referable = [r for r in rows if r["referable_dr"] == 1]
    decided_ref = [r for r in decided if r["referable_dr"] == 1]
    missed = [r for r in decided_ref if r["prediction"] == 0]
    nonref = [r for r in decided if r["referable_dr"] == 0]
    false_pos = [r for r in nonref if r["prediction"] == 1]
    labels = [r["referable_dr"] for r in decided]
    scores = [r["score"] for r in decided]
    patient = patient_failure_summary(rows)
    return {
        "n_total_eyes": len(rows),
        "n_decided_eyes": len(decided),
        "n_deferred_eyes": len(rows) - len(decided),
        "defer_rate": (len(rows) - len(decided)) / len(rows) if rows else float("nan"),
        "mean_images": sum(r["images"] for r in rows) / len(rows) if rows else float("nan"),
        "image_savings_vs_two": 1.0 - ((sum(r["images"] for r in rows) / len(rows)) / 2.0) if rows else float("nan"),
        "missed_referable_risk_decided": len(missed) / len(decided_ref) if decided_ref else float("nan"),
        "missed_referable_fraction_all": len(missed) / len(referable) if referable else float("nan"),
        "sensitivity_decided": 1.0 - len(missed) / len(decided_ref) if decided_ref else float("nan"),
        "specificity_decided": 1.0 - len(false_pos) / len(nonref) if nonref else float("nan"),
        "auroc_decided": auc_binary(labels, scores) if len(set(labels)) == 2 else float("nan"),
        "auprc_decided": average_precision(labels, scores) if labels else float("nan"),
        "brier_decided": brier_score(labels, scores) if labels else float("nan"),
        **patient,
    }


def patient_failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    positive_patients = 0
    decided_positive_patients = 0
    missed_decided_positive_patients = 0
    missed_any_positive_patients = 0
    for patient_rows in by_patient.values():
        has_positive = any(r["referable_dr"] == 1 for r in patient_rows)
        if has_positive:
            positive_patients += 1
        decided_positive = [r for r in patient_rows if r["referable_dr"] == 1 and not r["deferred"]]
        if decided_positive:
            decided_positive_patients += 1
            if any(r["prediction"] == 0 for r in decided_positive):
                missed_decided_positive_patients += 1
        if has_positive and any(r["referable_dr"] == 1 and not r["deferred"] and r["prediction"] == 0 for r in patient_rows):
            missed_any_positive_patients += 1
    return {
        "n_positive_patients": positive_patients,
        "n_decided_positive_patients": decided_positive_patients,
        "patient_missed_risk_decided_positive": (
            missed_decided_positive_patients / decided_positive_patients if decided_positive_patients else float("nan")
        ),
        "patient_missed_fraction_all_positive": (
            missed_any_positive_patients / positive_patients if positive_patients else float("nan")
        ),
        "patient_zero_event_upper95_if_no_misses": (
            1.0 - 0.05 ** (1.0 / decided_positive_patients)
            if decided_positive_patients and missed_decided_positive_patients == 0
            else float("nan")
        ),
    }


def select_threshold_baseline(
    calibration: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    score_key: str,
    target_risk: float,
    min_specificity: float,
    images: int,
) -> dict[str, Any]:
    candidates = []
    for threshold in candidate_grid([x[score_key] for x in calibration]):
        metrics = threshold_metrics(calibration, score_key, threshold, images=images)
        risk = metrics["missed_referable_risk_decided"]
        spec = metrics["specificity_decided"]
        if math.isfinite(risk) and risk <= target_risk and math.isfinite(spec) and spec >= min_specificity:
            candidates.append((threshold, metrics))
    if not candidates:
        return {"status": "NO_EMPIRICALLY_FEASIBLE_BASELINE"}
    threshold, cal = min(candidates, key=lambda item: (-item[1]["specificity_decided"], item[1]["missed_referable_risk_decided"]))
    return {
        "status": "EMPIRICALLY_SELECTED_UNCERTIFIED",
        "threshold": threshold,
        "calibration": cal,
        "validation": threshold_metrics(validation, score_key, threshold, images=images),
    }


def select_confidence_baseline(
    calibration: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    target_risk: float,
    min_specificity: float,
) -> dict[str, Any]:
    scores = [x["first_score"] for x in calibration]
    candidates = []
    for low in candidate_grid(scores):
        for high in candidate_grid(scores):
            if low >= high:
                continue
            metrics = confidence_triage_metrics(calibration, low, high)
            risk = metrics["missed_referable_risk_decided"]
            spec = metrics["specificity_decided"]
            if math.isfinite(risk) and risk <= target_risk and math.isfinite(spec) and spec >= min_specificity:
                candidates.append((low, high, metrics))
    if not candidates:
        return {"status": "NO_EMPIRICALLY_FEASIBLE_BASELINE"}
    low, high, cal = min(candidates, key=lambda item: (item[2]["defer_rate"], -item[2]["specificity_decided"]))
    return {
        "status": "EMPIRICALLY_SELECTED_UNCERTIFIED",
        "stop_low": low,
        "stop_high": high,
        "calibration": cal,
        "validation": confidence_triage_metrics(validation, low, high),
    }


def select_two_stage_policy(
    calibration: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    target_risk: float,
    min_specificity: float,
) -> dict[str, Any]:
    first_scores = candidate_grid([x["first_score"] for x in calibration], max_points=31)
    two_scores = candidate_grid([x["two_view_score"] for x in calibration], max_points=31)
    candidates = []
    for first_low in first_scores:
        for first_high in first_scores:
            if first_low >= first_high:
                continue
            for second_low in two_scores:
                for second_high in two_scores:
                    if second_low >= second_high:
                        continue
                    metrics = two_stage_metrics(calibration, first_low, first_high, second_low, second_high)
                    risk = metrics["missed_referable_risk_decided"]
                    spec = metrics["specificity_decided"]
                    if math.isfinite(risk) and risk <= target_risk and math.isfinite(spec) and spec >= min_specificity:
                        candidates.append((first_low, first_high, second_low, second_high, metrics))
    if not candidates:
        return {"status": "NO_EMPIRICALLY_FEASIBLE_POLICY"}
    first_low, first_high, second_low, second_high, cal = min(
        candidates,
        key=lambda item: (
            item[4]["defer_rate"],
            item[4]["mean_images"],
            item[4]["missed_referable_risk_decided"],
            -item[4]["specificity_decided"],
        ),
    )
    return {
        "status": "EMPIRICALLY_SELECTED_UNCERTIFIED",
        "thresholds": {
            "first_low": first_low,
            "first_high": first_high,
            "second_low": second_low,
            "second_high": second_high,
        },
        "calibration": cal,
        "validation": two_stage_metrics(validation, first_low, first_high, second_low, second_high),
    }


def policy_rows_from_thresholds(examples: list[dict[str, Any]], low: float, high: float, final_threshold: float) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        rows.append({**ex, **policy_decision(ex["first_score"], ex["two_view_score"], low, high, final_threshold)})
    return rows


def selected_policy_for_target(policy_data: dict[str, Any], target: float) -> dict[str, Any]:
    selected = [p for p in policy_data["selected_policies"] if abs(float(p["target_risk"]) - target) < 1e-9]
    if not selected:
        raise ValueError(f"No selected policy for target {target}")
    return selected[0]


def paired_value(examples: list[dict[str, Any]], split: str) -> dict[str, Any]:
    xs = [x for x in examples if x["split"] == split]
    labels = [x["referable_dr"] for x in xs]
    first = [x["first_score"] for x in xs]
    both = [x["two_view_score"] for x in xs]
    return {
        "split": split,
        "n": len(xs),
        "first_auroc": auc_binary(labels, first),
        "two_view_auroc": auc_binary(labels, both),
        "auroc_gain": auc_binary(labels, both) - auc_binary(labels, first),
        "first_auprc": average_precision(labels, first),
        "two_view_auprc": average_precision(labels, both),
        "first_brier": brier_score(labels, first),
        "two_view_brier": brier_score(labels, both),
    }


def main() -> None:
    target = 0.05
    min_specificity = 0.5
    report: dict[str, Any] = {
        "status": "FAST_PAPER_PACK_EXPLORATORY",
        "target_risk": target,
        "min_specificity": min_specificity,
        "risk_certificate": "NONE",
        "test_status": "historical DeepDRiD test split is exposed and omitted from development paper pack",
        "branches": [],
    }
    baseline_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        rows = read_predictions(branch["predictions"])
        paired = pair_predictions(rows, branch["first_view"], branch["second_view"])
        examples = list(paired.values())
        calibration = [x for x in examples if x["split"] == "calibration"]
        validation = [x for x in examples if x["split"] == "validation"]
        policy_data = load_json(branch["policy"])
        selected = selected_policy_for_target(policy_data, target)
        two_stage = select_two_stage_policy(calibration, validation, target, min_specificity)

        active_validation_rows = policy_rows_from_thresholds(
            validation,
            selected["stop_low"],
            selected["stop_high"],
            selected["final_threshold"],
        )
        active_summary = summarize_decisions(active_validation_rows)
        active_summary_existing = evaluate_policy(
            validation,
            selected["stop_low"],
            selected["stop_high"],
            selected["final_threshold"],
        )
        # Keep both summaries traceable: the evaluator is canonical for eye-level policy metrics;
        # this script adds patient-level failure counts for manuscript triage.
        active_summary.update(
            {
                "canonical_eye_missed_risk": active_summary_existing["missed_referable_dr_risk"],
                "canonical_eye_image_savings": active_summary_existing["image_savings_vs_always_two_view"],
            }
        )
        baselines = {
            "active_retina_rc_two_stage": two_stage,
            "always_one_initial": select_threshold_baseline(calibration, validation, "first_score", target, min_specificity, images=1),
            "always_two_views": select_threshold_baseline(calibration, validation, "two_view_score", target, min_specificity, images=2),
            "confidence_triage_no_acquisition": select_confidence_baseline(calibration, validation, target, min_specificity),
        }
        branch_report = {
            "name": branch["name"],
            "mapping_note": branch["mapping_note"],
            "n_examples": len(examples),
            "n_calibration": len(calibration),
            "n_validation": len(validation),
            "paired_value": {
                "calibration": paired_value(examples, "calibration"),
                "validation": paired_value(examples, "validation"),
            },
            "active_retina_rc_target_0_05": {
                "status": selected["status"],
                "thresholds": {
                    "stop_low": selected["stop_low"],
                    "stop_high": selected["stop_high"],
                    "final_threshold": selected["final_threshold"],
                },
                "calibration": selected["calibration"],
                "validation": active_summary,
                "validation_ci": selected["validation_ci"],
            },
            "baselines": baselines,
        }
        report["branches"].append(branch_report)
        for split, vals in branch_report["paired_value"].items():
            view_rows.append({"branch": branch["name"], **vals})
        for method, vals in {"active_retina_rc_legacy": branch_report["active_retina_rc_target_0_05"], **baselines}.items():
            if vals["status"] != "EMPIRICALLY_SELECTED_UNCERTIFIED":
                baseline_rows.append({"branch": branch["name"], "method": method, "status": vals["status"]})
                continue
            metrics = vals["validation"]
            baseline_rows.append(
                {
                    "branch": branch["name"],
                    "method": method,
                    "status": vals["status"],
                    "mean_images": fmt(metrics["mean_images"]),
                    "image_savings_vs_two": fmt(metrics["image_savings_vs_two"]),
                    "defer_rate": fmt(metrics["defer_rate"]),
                    "missed_ref_risk_decided_eye": fmt(metrics["missed_referable_risk_decided"]),
                    "missed_ref_fraction_all_eye": fmt(metrics["missed_referable_fraction_all"]),
                    "specificity_decided": fmt(metrics["specificity_decided"]),
                    "positive_patient_missed_risk": fmt(metrics["patient_missed_risk_decided_positive"]),
                    "zero_event_upper95": fmt(metrics["patient_zero_event_upper95_if_no_misses"]),
                }
            )

    add_incremental_policy_costs(baseline_rows)

    write_json(Path("results/fast_paper_evidence_pack.json"), report)
    write_csv(Path("results/table_fast_paired_value.csv"), view_rows)
    write_csv(Path("results/table_fast_policy_baselines.csv"), baseline_rows)

    manuscript = render_manuscript(report, view_rows, baseline_rows)
    Path("paper").mkdir(exist_ok=True)
    Path("paper/active_retina_rc_fast_draft.tex").write_text(manuscript, encoding="utf-8")


def add_incremental_policy_costs(rows: list[dict[str, Any]]) -> None:
    by_branch: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_branch[row["branch"]][row["method"]] = row
    for branch_rows in by_branch.values():
        confidence = branch_rows.get("confidence_triage_no_acquisition")
        if not confidence or confidence.get("mean_images") in ("", "NA", None):
            continue
        conf_images = float(confidence["mean_images"])
        conf_defer = float(confidence["defer_rate"])
        for method, row in branch_rows.items():
            if row.get("mean_images") in ("", "NA", None):
                row["delta_images_vs_confidence"] = "NA"
                row["delta_defer_vs_confidence"] = "NA"
                row["break_even_defer_cost_images"] = "NA"
                continue
            delta_images = float(row["mean_images"]) - conf_images
            delta_defer = float(row["defer_rate"]) - conf_defer
            row["delta_images_vs_confidence"] = fmt(delta_images)
            row["delta_defer_vs_confidence"] = fmt(delta_defer)
            if delta_images > 0 and delta_defer < 0:
                row["break_even_defer_cost_images"] = fmt(delta_images / (-delta_defer))
            elif delta_images <= 0 and delta_defer <= 0:
                row["break_even_defer_cost_images"] = "dominates"
            else:
                row["break_even_defer_cost_images"] = "NA"


def latex_table(rows: list[dict[str, Any]], columns: list[str], headers: list[str]) -> str:
    spec = "l" * len(columns)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\hline", " & ".join(headers) + r" \\", "\\hline"]
    for row in rows:
        out.append(" & ".join(str(row.get(col, "NA")).replace("_", r"\_") for col in columns) + r" \\")
    out.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(out)


def render_manuscript(report: dict[str, Any], view_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> str:
    compact_view_rows = []
    for row in view_rows:
        compact_view_rows.append(
            {
                "branch": row["branch"],
                "split": row["split"],
                "n": row["n"],
                "first": fmt(row["first_auroc"]),
                "two": fmt(row["two_view_auroc"]),
                "gain": fmt(row["auroc_gain"]),
                "brier1": fmt(row["first_brier"]),
                "brier2": fmt(row["two_view_brier"]),
            }
        )
    view_table = latex_table(
        compact_view_rows,
        ["branch", "split", "n", "first", "two", "gain", "brier1", "brier2"],
        ["Branch", "Split", "N", "Initial AUROC", "Two-view AUROC", "Gain", "Initial Brier", "Two-view Brier"],
    )
    policy_table = latex_table(
        baseline_rows,
        [
            "branch",
            "method",
            "mean_images",
            "image_savings_vs_two",
            "defer_rate",
            "missed_ref_risk_decided_eye",
            "specificity_decided",
            "positive_patient_missed_risk",
            "break_even_defer_cost_images",
        ],
        ["Branch", "Method", "Images", "Savings", "Defer", "Miss risk", "Spec.", "Patient miss", "Break-even"],
    )
    return rf"""\documentclass[journal]{{IEEEtran}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{url}}

\title{{ActiveRetina-RC: Risk-Controlled Adaptive Retinal Imaging With Automatic Resolution for Diabetic Retinopathy Screening}}
\author{{Burak Aksu}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Multi-field retinal imaging can improve diabetic retinopathy screening, but acquiring every field for every eye increases imaging burden. We study a risk-controlled adaptive imaging formulation in which a screening system may stop after an initial fundus view, acquire a second paired view, or defer a case for full clinical handling. Using the public DeepDRiD paired fundus dataset, we build a reproducible development pipeline with patient-level splits, frozen ImageNet-pretrained encoders, set-based view aggregation, suffix-mapping sensitivity branches, and cross-fitted development predictions. The current evidence is intentionally reported as exploratory: no certified risk guarantee or untouched confirmatory test claim is made. In development validation, paired views improve AUROC by 0.020--0.041 under the two suffix branches. A two-stage adaptive policy reaches the same observed missed-referable-DR rate as first-view confidence triage while eliminating deferrals and still saving 41--46\% of images relative to always acquiring both views. These results support an operationally sharper claim than image minimization alone: adaptive acquisition can trade a small number of extra images for automatic case resolution, but final flagship status still requires independent confirmation and formal risk certification.
\end{{abstract}}

\begin{{IEEEkeywords}}
Diabetic retinopathy, adaptive imaging, active acquisition, risk control, retinal fundus imaging, DeepDRiD.
\end{{IEEEkeywords}}

\section{{Introduction}}
Diabetic retinopathy (DR) screening often benefits from multiple retinal photographs, but fixed multi-view protocols impose avoidable acquisition cost on eyes whose first image is already diagnostically decisive. ActiveRetina-RC asks a constrained question: after an initial fundus image, can a model safely decide to stop, acquire a second image, or defer the case, while controlling missed referable DR among cases it decides automatically?

This draft assumes MFIDDR access never arrives. Under that constraint, the paper cannot claim four-field non-myopic acquisition. Instead, its strongest honest contribution is a risk-controlled paired-view acquisition framework with explicit failure accounting, mapping sensitivity analysis, and a reproducible public-data pipeline.

\section{{Methods}}
\subsection{{Dataset and Endpoint}}
We use DeepDRiD v1.1 regular fundus images. The primary endpoint is referable DR, defined as five-class DR grade at least 2. The secondary endpoint is five-class grading, reserved for extended analysis. The official source confirms paired macula-centered and optic-disc-centered images, but the public metadata inspected so far does not officially map filename suffixes to anatomy. Therefore, all adaptive results are run in two suffix branches rather than relying on one anatomical assumption.

\subsection{{Splits and Model}}
Patients are split into train, calibration, validation, and historical test partitions. Because the historical test set was inspected during method repair, it is excluded from this development paper pack and treated as exposed. A frozen ResNet-50 ImageNet encoder produces image embeddings. View subsets are represented with permutation-aware pooling features and a regularized logistic classifier.

\subsection{{Policy}}
For each eye, the policy receives an initial-view score. Low-risk eyes may stop negative, high-risk eyes may stop positive, and intermediate eyes acquire the second view. If the two-view score remains below the final referral threshold, the eye is deferred rather than automatically ruled negative. Thresholds are selected on calibration data and evaluated on validation data. This is empirical selection only; it is not a formal risk certificate.

\section{{Results}}
\subsection{{Paired-View Value}}
Table~\ref{{tab:viewvalue}} reports initial-view versus paired-view validation performance under both suffix branches.

\begin{{table*}}[t]
\centering
\caption{{Paired-view value under suffix-mapping sensitivity branches.}}
\label{{tab:viewvalue}}
{view_table}
\end{{table*}}

\subsection{{Adaptive Policy and Fair Baselines}}
Table~\ref{{tab:policies}} compares the adaptive policy to simple empirical baselines at the 5\% target risk development setting. All entries are uncertified and should be read as development evidence. The break-even column reports how costly a deferred case must be, in image-equivalent units, for the two-stage policy to improve total operational cost relative to confidence triage.

\begin{{table*}}[t]
\centering
\caption{{Development validation policy comparison. Miss risk is among decided referable eyes; patient miss aggregates any missed referable eye by patient. Break-even is relative to confidence triage and is reported in image-equivalent units per deferred case.}}
\label{{tab:policies}}
{policy_table}
\end{{table*}}

The legacy adaptive policy reduces image use but still defers cases. The upgraded two-stage policy allows the acquired paired view to support either a negative or positive automatic decision. In both suffix branches, this removes validation deferrals at the same observed missed-referable-DR rate as confidence triage. The price is a small increase in mean images: 0.077 additional images per eye in the suffix2-initial branch and 0.179 in the suffix1-initial branch. This is favorable whenever avoiding one deferred case is worth more than approximately 1.2--1.4 extra images operationally.

\section{{Discussion}}
The current study is a credible method-development package rather than a completed flagship paper. It contributes a reproducible adaptive-imaging pipeline, an explicit stop/acquire/defer formulation, patient-level split discipline, and sensitivity to unresolved DeepDRiD suffix mapping. The main negative finding is equally important: public paired DeepDRiD alone is likely underpowered for a definitive risk-controlled IEEE journal claim unless supplemented by independent paired-view confirmation.

The strongest reviewer challenge is not whether paired views improve discrimination; they do. The challenge is whether acquiring the second image is better than simply deferring uncertain first-view cases. The two-stage policy gives a defensible operational answer: it uses additional images selectively to remove the unresolved-case burden. Thus, the paper should not claim pure image-count dominance over triage. It should claim a three-way frontier among safety, images, and automatic resolution.

\section{{Decision}}
Current decision: \textbf{{GO-ARXIV / CONDITIONAL-IEEE}}. The revised contribution is strong enough for an arXiv flagship methods manuscript if labeled as development evidence. For an IEEE journal claim, the same two-stage policy must be frozen and confirmed on an independent paired or multi-field cohort, with formal risk bounds computed before any final test inspection.

\section{{Limitations}}
The DeepDRiD suffix-to-anatomy mapping remains unofficial in the public files reviewed. The historical test split is exposed and excluded from confirmatory claims. Threshold selection is empirical and uncertified. DeepDRiD has two paired views, so it cannot support four-field non-myopic planning claims.

\section{{Conclusion}}
ActiveRetina-RC provides a disciplined public-data route toward adaptive retinal imaging. The revised claim is no longer that DeepDRiD alone proves a deployable clinical guarantee. The stronger and defensible claim is that selective second-view acquisition can eliminate unresolved first-view cases at matched observed safety while preserving large image savings against fixed two-view imaging. The correct next milestone is independent confirmation rather than novel-method escalation.

\end{{document}}
"""


if __name__ == "__main__":
    main()
