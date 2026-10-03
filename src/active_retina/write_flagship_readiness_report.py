from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.4f}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_lines(name: str, data: dict) -> list[str]:
    return [
        f"## {name} Paired-View Value",
        "",
        f"- Status: `{data['status']}`",
        f"- Mapping status: `{data['view_mapping_status']}`",
        f"- Paired eyes: {data['n_paired_eyes']}",
        f"- First-view AUROC: {fmt(data['single_first_view']['auroc'])}; Brier: {fmt(data['single_first_view']['brier'])}",
        f"- Two-view AUROC: {fmt(data['two_view']['auroc'])}; Brier: {fmt(data['two_view']['brier'])}",
        f"- AUROC gain from second view: {fmt(data['two_view']['auroc'] - data['single_first_view']['auroc'])}",
        f"- Cross-fitted hindsight AUROC upper bound: {fmt(data['cross_fitted_hindsight_stop_oracle']['auroc'])}",
        f"- Fraction of eyes where second view reduced Brier loss: {fmt(data['cross_fitted_hindsight_stop_oracle']['fraction_eyes_second_view_helped'])}",
        "",
    ]


def policy_lines(name: str, data: dict) -> list[str]:
    lines = [
        f"## {name} Calibrated Policy",
        "",
        f"- Status: `{data['status']}`",
        f"- Decision: `{data['decision']}`",
        f"- Calibration eyes: {data['n_calibration']}",
        f"- Validation eyes: {data['n_validation']}",
        f"- Sealed-test eyes: {data['n_sealed_test']}",
        "",
        "| Target risk | Calibration status | Sealed-test missed-risk | Mean images | Savings vs two-view | Defer rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in data["selected_policies"]:
        if row["status"] != "CERTIFIED_ON_CALIBRATION":
            lines.append(f"| {row['target_risk']:.2f} | `{row['status']}` | | | | |")
            continue
        test = row["sealed_test"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['target_risk']:.2f}",
                    f"`{row['status']}`",
                    fmt(test["missed_referable_dr_risk"]),
                    fmt(test["mean_images"]),
                    fmt(test["image_savings_vs_always_two_view"]),
                    fmt(test["defer_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/flagship_readiness_report.md")
    args = parser.parse_args()

    visual = load(Path("results/deepdrid_active_retina_rc_metrics.json"))
    swapped = load(Path("results/deepdrid_swapped_mapping_active_retina_rc_metrics.json"))
    visual_policy = load(Path("results/deepdrid_active_retina_rc_metrics_policy.json"))
    swapped_policy = load(Path("results/deepdrid_swapped_mapping_active_retina_rc_metrics_policy.json"))

    lines = [
        "# ActiveRetina-RC Flagship Readiness Report",
        "",
        "Date: 2026-10-03",
        "",
        "Status: `PIPELINE_READY_BASELINE_EVIDENCE_AVAILABLE`",
        "",
        "This report summarizes the executable baseline produced after the flagship upgrade. The current model family is a dependency-light image-feature logistic baseline, used to verify the end-to-end pipeline and provide preliminary evidence. It is not the final pretrained encoder needed for paper-level performance claims.",
        "",
        "## Current Decision",
        "",
        "`PRELIMINARY_PIPELINE_GO`",
        "",
        "The pipeline is ready for the stronger pretrained-encoder run. The preliminary baseline shows paired-view value and viable calibrated policy behavior under both DeepDRiD mapping assumptions, but final flagship claims remain pending a stronger encoder and frozen final reporting.",
        "",
    ]
    lines += paired_lines("Visual-Audit Mapping", visual)
    lines += paired_lines("Swapped Mapping Sensitivity", swapped)
    lines += policy_lines("Visual-Audit Mapping", visual_policy)
    lines += policy_lines("Swapped Mapping Sensitivity", swapped_policy)
    lines += [
        "## What Is Ready",
        "",
        "- DeepDRiD local data audit and immutable patient splits.",
        "- Mapping sensitivity branch.",
        "- Cross-fitted prediction CSV generation.",
        "- Paired-view value evaluator.",
        "- Calibrated `STOP / ACQUIRE_SECOND_VIEW / DEFER` policy evaluator.",
        "- Bootstrap CIs in metrics JSON outputs.",
        "- Explicit evidence boundaries separating executable baseline evidence from final pretrained-encoder claims.",
        "",
        "## Still Required For The Actual Paper",
        "",
        "- Replace the dependency-light baseline with a strong pretrained retinal/image encoder.",
        "- Re-run both mapping branches with the stronger encoder.",
        "- Add external public DR robustness datasets where legally accessible.",
        "- Freeze policy families before final sealed-test reporting.",
        "- Generate publication plots and IEEEtran manuscript tables from frozen outputs.",
        "",
        "## Evidence Boundary",
        "",
        "No MFIDDR, four-field CCG, non-myopic planning, clinical deployment, or official DeepDRiD suffix-mapping claim is made here.",
        "",
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
