from __future__ import annotations

import csv
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.4f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paired_specs = [
        ("visual_audit", load("results/deepdrid_pretrained_active_retina_rc_metrics.json")),
        ("swapped_sensitivity", load("results/deepdrid_swapped_mapping_pretrained_active_retina_rc_metrics.json")),
    ]
    paired_rows = []
    for branch, data in paired_specs:
        paired_rows.append(
            {
                "mapping_branch": branch,
                "mapping_status": data["view_mapping_status"],
                "n_paired_eyes": data["n_paired_eyes"],
                "first_view_auroc": data["single_first_view"]["auroc"],
                "two_view_auroc": data["two_view"]["auroc"],
                "auroc_gain": data["two_view"]["auroc"] - data["single_first_view"]["auroc"],
                "first_view_brier": data["single_first_view"]["brier"],
                "two_view_brier": data["two_view"]["brier"],
                "hindsight_oracle_auroc": data["cross_fitted_hindsight_stop_oracle"]["auroc"],
                "fraction_second_view_helped": data["cross_fitted_hindsight_stop_oracle"]["fraction_eyes_second_view_helped"],
            }
        )
    write_csv(Path("results/table_pretrained_paired_view_value.csv"), paired_rows)

    policy_specs = [
        ("visual_audit", load("results/deepdrid_pretrained_active_retina_rc_metrics_policy.json")),
        ("swapped_sensitivity", load("results/deepdrid_swapped_mapping_pretrained_active_retina_rc_metrics_policy.json")),
    ]
    policy_rows = []
    decisions = {}
    for branch, data in policy_specs:
        decisions[branch] = data["decision"]
        for row in data["selected_policies"]:
            base = {
                "mapping_branch": branch,
                "target_risk": row["target_risk"],
                "status": row["status"],
                "min_specificity": row.get("min_specificity", data.get("min_specificity_constraint")),
            }
            if row["status"] == "CERTIFIED_ON_CALIBRATION":
                test = row["sealed_test"]
                base.update(
                    {
                        "sealed_test_missed_risk": test["missed_referable_dr_risk"],
                        "mean_images": test["mean_images"],
                        "image_savings_vs_always_two_view": test["image_savings_vs_always_two_view"],
                        "defer_rate": test["defer_rate"],
                        "sensitivity_among_decided": test["sensitivity_among_decided"],
                        "specificity_among_decided": test["specificity_among_decided"],
                    }
                )
            policy_rows.append(base)
    write_csv(Path("results/table_pretrained_policy_cost_risk.csv"), policy_rows)

    overall = "METHOD_ONLY_WEAK_EVIDENCE_MAPPING_SENSITIVITY_BLOCKER"
    if decisions.get("visual_audit") == "PRELIMINARY_PIPELINE_GO" and decisions.get("swapped_sensitivity") == "PRELIMINARY_PIPELINE_GO":
        overall = "PRELIMINARY_STRONG_ENCODER_GO"

    lines = [
        "# ActiveRetina-RC Pretrained Encoder Readiness Report",
        "",
        "Date: 2026-10-03",
        "",
        f"Status: `{overall}`",
        "",
        "Model: frozen ImageNet-pretrained `resnet50.a1_in1k` features with regularized logistic heads.",
        "",
        "This is stronger than the dependency-light baseline and is now the main DeepDRiD evidence artifact. It is still not a completed paper package because external robustness datasets and final manuscript figures remain pending.",
        "",
        "## Paired-View Value",
        "",
        "| Mapping branch | First-view AUROC | Two-view AUROC | AUROC gain | First Brier | Two-view Brier | Hindsight AUROC | Second view helped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mapping_branch"],
                    fmt(row["first_view_auroc"]),
                    fmt(row["two_view_auroc"]),
                    fmt(row["auroc_gain"]),
                    fmt(row["first_view_brier"]),
                    fmt(row["two_view_brier"]),
                    fmt(row["hindsight_oracle_auroc"]),
                    fmt(row["fraction_second_view_helped"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Calibrated Policy With Specificity Floor",
        "",
        "| Mapping branch | Target risk | Status | Test missed-risk | Mean images | Savings | Defer rate | Sensitivity | Specificity |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        if row["status"] != "CERTIFIED_ON_CALIBRATION":
            lines.append(f"| {row['mapping_branch']} | {row['target_risk']:.2f} | `{row['status']}` | | | | | | |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mapping_branch"],
                    f"{float(row['target_risk']):.2f}",
                    f"`{row['status']}`",
                    fmt(row["sealed_test_missed_risk"]),
                    fmt(row["mean_images"]),
                    fmt(row["image_savings_vs_always_two_view"]),
                    fmt(row["defer_rate"]),
                    fmt(row["sensitivity_among_decided"]),
                    fmt(row["specificity_among_decided"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Pretrained features produce strong paired-view discrimination: two-view AUROC is about 0.904 in both mapping branches.",
        "- The visual-audit branch supports a preliminary image-saving policy under the 0.10 target risk with a 0.50 specificity floor.",
        "- The swapped branch fails the same sealed-test risk check under the specificity-constrained policy, so suffix mapping remains a material blocker for a field-named paper claim.",
        "- A suffix-agnostic paired-view claim is possible, but the stronger IEEE/arXiv paper still needs mapping resolution or a mapping-robust policy family.",
        "",
        "## Next Required Work",
        "",
        "- Resolve DeepDRiD suffix mapping by official source or make the paper explicitly suffix-agnostic.",
        "- Add at least one external public DR robustness dataset.",
        "- Generate final plots and IEEEtran manuscript tables after the policy family is frozen.",
        "",
    ]
    Path("docs/pretrained_encoder_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
