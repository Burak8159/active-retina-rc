"""Summarize DDR autonomy recovery runs into tables and figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "resnet50_frozen": "results/ddr_open_autonomy_specificity100/summary.json",
    "convnext_tiny_frozen": "results/ddr_open_autonomy_specificity100_convnext_tiny/summary.json",
    "dino_vit_frozen": "results/ddr_open_autonomy_specificity100_vit_dino/summary.json",
    "ensemble_frozen": "results/ddr_open_autonomy_specificity100_ensemble/summary.json",
    "ensemble_observed_threshold_diagnostic": "results/ddr_open_autonomy_specificity100_ensemble_observed_thresholds/summary.json",
    "resnet50_finetune_e2": "results/ddr_finetuned_autonomy_resnet50_e2/summary.json",
    "feature_mlp_h1024": "results/rtx3050_feature_mlp_ensemble_h1024/summary.json",
    "feature_mlp_h512_d60_wd1e3": "results/rtx3050_feature_mlp_ensemble_h512_d60_wd1e3/summary.json",
    "specificity300_resnet50": "results/ddr_open_autonomy_specificity300/summary.json",
    "specificity300_resnet50_observed_threshold_diagnostic": "results/ddr_open_autonomy_specificity300_observed_thresholds/summary.json",
    "specificity300_dino_vit": "results/ddr_open_autonomy_specificity300_vit_dino/summary.json",
    "specificity300_convnext_tiny": "results/ddr_open_autonomy_specificity300_convnext_tiny/summary.json",
    "specificity300_resnet_dino": "results/ddr_open_autonomy_specificity300_resnet_dino/summary.json",
    "specificity300_three_encoder_ensemble": "results/ddr_open_autonomy_specificity300_ensemble/summary.json",
    "specificity300_dino_mlp": "results/ddr_feature_mlp_specificity300_dino_h512_d50/summary.json",
    "specificity300_ensemble_mlp": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50/summary.json",
    "specificity300_ensemble_mlp_conservative": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50_conservative/summary.json",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_name(summary: dict[str, Any]) -> str:
    if "best_candidate" in summary:
        return str(summary["best_candidate"].get("candidate", "candidate"))
    return "trained_artifact"


def split_metrics(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    if "best_candidate" in summary:
        cand = summary["best_candidate"]
        cert = summary.get(f"{prefix}_certificate", {})
        return {
            f"{prefix}_auroc": cand.get(f"{prefix}_auroc"),
            f"{prefix}_average_precision": cand.get(f"{prefix}_average_precision"),
            f"{prefix}_specificity": cert.get("specificity", cand.get(f"{prefix}_specificity")),
            f"{prefix}_missed_positive": cert.get("missed_positive", cand.get(f"{prefix}_missed_positive")),
            f"{prefix}_positive": cert.get("positive", summary.get(f"{prefix}_positive")),
            f"{prefix}_missed_risk_ucb95": cert.get("missed_risk_ucb95", cand.get(f"{prefix}_ucb95")),
            f"{prefix}_referable_action_rate": cert.get("referable_action_rate", cand.get(f"{prefix}_referable_action_rate")),
            f"{prefix}_precision_referable_action": cert.get("precision_referable_action", cand.get(f"{prefix}_precision_referable_action")),
        }
    cert = summary.get(f"{prefix}_certificate", {})
    return {
        f"{prefix}_auroc": summary.get(f"{prefix}_auroc"),
        f"{prefix}_average_precision": summary.get(f"{prefix}_average_precision"),
        f"{prefix}_specificity": cert.get("specificity"),
        f"{prefix}_missed_positive": cert.get("missed_positive"),
        f"{prefix}_positive": cert.get("positive"),
        f"{prefix}_missed_risk_ucb95": cert.get("missed_risk_ucb95"),
        f"{prefix}_referable_action_rate": cert.get("referable_action_rate"),
        f"{prefix}_precision_referable_action": cert.get("precision_referable_action"),
    }


def build_table(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, rel_path in DEFAULT_RUNS.items():
        summary = load_json(project_root / rel_path)
        if summary is None:
            continue
        row = {
            "run": run_name,
            "candidate": candidate_name(summary),
            "status": summary.get("status"),
            "source": rel_path,
        }
        row.update(split_metrics(summary, "valid"))
        row.update(split_metrics(summary, "test"))
        row["test_certifies_5pct_95conf"] = bool(
            row.get("test_missed_positive") == 0
            and row.get("test_missed_risk_ucb95") is not None
            and float(row["test_missed_risk_ucb95"]) <= 0.05
        )
        row["meets_specificity70"] = bool(
            row["test_certifies_5pct_95conf"]
            and row.get("test_specificity") is not None
            and float(row["test_specificity"]) >= 0.70
        )
        rows.append(row)
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" text-anchor="{anchor}" fill="#1f2933">{safe}</text>'


def plot_specificity(rows: list[dict[str, Any]], out: Path) -> None:
    plot_rows = sorted(rows, key=lambda r: as_float(r.get("test_specificity")))
    width = 980
    left = 310
    right = 55
    top = 58
    row_h = 36
    height = top + row_h * len(plot_rows) + 70
    chart_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 30, "DDR autonomy specificity recovery", 18, weight="700"),
        svg_text(left, height - 22, "Test specificity at zero-miss threshold (%)", 13),
    ]
    for tick in range(0, 76, 10):
        x = left + chart_w * tick / 75
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 55}" stroke="#d7dde3" stroke-width="1"/>')
        parts.append(svg_text(x, height - 38, str(tick), 10, anchor="middle"))
    target_x = left + chart_w * 70 / 75
    parts.append(f'<line x1="{target_x:.1f}" y1="{top - 18}" x2="{target_x:.1f}" y2="{height - 55}" stroke="#222" stroke-dasharray="5 5" stroke-width="1.5"/>')
    parts.append(svg_text(target_x - 4, top - 26, "70% target", 11, anchor="end"))
    for i, row in enumerate(plot_rows):
        y = top + i * row_h
        spec = 100 * as_float(row.get("test_specificity"))
        bar_w = chart_w * min(spec, 75) / 75
        color = "#2f6f73" if row.get("test_certifies_5pct_95conf") else "#a33f3f"
        parts.append(svg_text(24, y + 18, str(row["run"]), 11))
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="22" fill="{color}" rx="3"/>')
        parts.append(svg_text(left + bar_w + 6, y + 16, f"{spec:.1f}%", 11))
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_risk_specificity(rows: list[dict[str, Any]], out: Path) -> None:
    width = 840
    height = 560
    left = 86
    right = 40
    top = 56
    bottom = 72
    chart_w = width - left - right
    chart_h = height - top - bottom
    x_max = 75.0
    y_max = max(6.0, max(100 * as_float(r.get("test_missed_risk_ucb95")) for r in rows) * 1.2)

    def x_pos(spec_pct: float) -> float:
        return left + chart_w * spec_pct / x_max

    def y_pos(risk_pct: float) -> float:
        return top + chart_h * (1 - risk_pct / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 30, "Risk-specificity frontier", 18, weight="700"),
        svg_text(width / 2, height - 24, "Test specificity (%)", 13, anchor="middle"),
        f'<text x="22" y="{height / 2:.1f}" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="#1f2933" transform="rotate(-90 22 {height / 2:.1f})" text-anchor="middle">95% upper missed-risk bound (%)</text>',
    ]
    for tick in range(0, 76, 10):
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#d7dde3" stroke-width="1"/>')
        parts.append(svg_text(x, height - bottom + 22, str(tick), 10, anchor="middle"))
    for tick in range(0, int(y_max) + 1):
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e7ebef" stroke-width="1"/>')
        parts.append(svg_text(left - 8, y + 4, str(tick), 10, anchor="end"))
    parts.append(f'<line x1="{left}" y1="{y_pos(5):.1f}" x2="{width - right}" y2="{y_pos(5):.1f}" stroke="#222" stroke-dasharray="5 5" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x_pos(70):.1f}" y1="{top}" x2="{x_pos(70):.1f}" y2="{height - bottom}" stroke="#222" stroke-dasharray="2 5" stroke-width="1.5"/>')
    for row in rows:
        spec = 100 * as_float(row.get("test_specificity"))
        risk = 100 * as_float(row.get("test_missed_risk_ucb95"))
        x = x_pos(spec)
        y = y_pos(risk)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#2f6f73"/>')
        parts.append(svg_text(x + 8, y - 8, str(row["run"]), 9))
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/ddr_autonomy_summary")
    args = parser.parse_args()

    root = Path.cwd()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = build_table(root)
    write_csv(out / "runs.csv", rows)
    if rows:
        plot_specificity(rows, out / "specificity_recovery.svg")
        plot_risk_specificity(rows, out / "risk_specificity_frontier.svg")
        best = sorted(rows, key=lambda r: as_float(r.get("test_specificity")), reverse=True)[0]
    else:
        best = None
    decision = {
        "status": "NO_GO_FULL_AUTONOMOUS_DIAGNOSIS",
        "best_specificity_run": best,
        "best_zero_miss_deployable_currently": "single-image autonomous referable action only; nonreferable clearance remains deferred",
        "specificity_target": 0.70,
        "output_table": str(out / "runs.csv"),
        "figures": [
            str(out / "specificity_recovery.svg"),
            str(out / "risk_specificity_frontier.svg"),
        ],
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
