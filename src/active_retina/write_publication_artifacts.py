from __future__ import annotations

import csv
import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_policy_plot(rows: list[dict], output: Path) -> None:
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 35, 65
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + x * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y) * plot_h

    colors = {"visual_audit": "#1f77b4", "swapped_sensitivity": "#d62728"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">Image savings vs always-two-view</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="14">Sealed-test missed referable DR risk</text>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="16">ActiveRetina-RC Baseline Cost-Risk Points</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<line x1="{sx(tick)}" y1="{top+plot_h}" x2="{sx(tick)}" y2="{top+plot_h+5}" stroke="#333"/>')
        parts.append(f'<text x="{sx(tick)}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.2f}</text>')
        parts.append(f'<line x1="{left-5}" y1="{sy(tick)}" x2="{left}" y2="{sy(tick)}" stroke="#333"/>')
        parts.append(f'<text x="{left-10}" y="{sy(tick)+4}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    for row in rows:
        x = float(row["image_savings_vs_always_two_view"])
        y = float(row["sealed_test_missed_risk"])
        color = colors[row["mapping_branch"]]
        parts.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{sx(x)+7}" y="{sy(y)-7}" font-family="Arial" font-size="10">{row["mapping_branch"]}, target {float(row["target_risk"]):.2f}</text>'
        )
    parts.append(f'<circle cx="{width-210}" cy="48" r="5" fill="{colors["visual_audit"]}"/>')
    parts.append(f'<text x="{width-198}" y="52" font-family="Arial" font-size="12">visual audit</text>')
    parts.append(f'<circle cx="{width-210}" cy="68" r="5" fill="{colors["swapped_sensitivity"]}"/>')
    parts.append(f'<text x="{width-198}" y="72" font-family="Arial" font-size="12">swapped sensitivity</text>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    paired = [
        ("visual_audit", load("results/deepdrid_active_retina_rc_metrics.json")),
        ("swapped_sensitivity", load("results/deepdrid_swapped_mapping_active_retina_rc_metrics.json")),
    ]
    paired_rows = []
    for branch, data in paired:
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
    write_csv(Path("results/table_paired_view_value.csv"), paired_rows)

    policies = [
        ("visual_audit", load("results/deepdrid_active_retina_rc_metrics_policy.json")),
        ("swapped_sensitivity", load("results/deepdrid_swapped_mapping_active_retina_rc_metrics_policy.json")),
    ]
    policy_rows = []
    for branch, data in policies:
        for row in data["selected_policies"]:
            if row["status"] != "CERTIFIED_ON_CALIBRATION":
                continue
            test = row["sealed_test"]
            policy_rows.append(
                {
                    "mapping_branch": branch,
                    "target_risk": row["target_risk"],
                    "status": row["status"],
                    "sealed_test_missed_risk": test["missed_referable_dr_risk"],
                    "mean_images": test["mean_images"],
                    "image_savings_vs_always_two_view": test["image_savings_vs_always_two_view"],
                    "defer_rate": test["defer_rate"],
                    "sensitivity_among_decided": test["sensitivity_among_decided"],
                    "specificity_among_decided": test["specificity_among_decided"],
                }
            )
    write_csv(Path("results/table_policy_cost_risk.csv"), policy_rows)
    svg_policy_plot(policy_rows, Path("plots/policy_cost_risk.svg"))


if __name__ == "__main__":
    main()
