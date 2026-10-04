"""Build paper-ready model-zoo tables and figures for DDR autonomy.

The goal is to mirror a mature biomedical ML evaluation figure: compare model
families, report balanced metrics, show score distributions, and make the
failure boundary explicit.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from active_retina.autonomy_failure_analysis import write_csv


RUNS = {
    "resnet50_frozen": "results/ddr_open_autonomy_specificity300/predictions.csv",
    "dino_vit_frozen": "results/ddr_open_autonomy_specificity300_vit_dino/predictions.csv",
    "convnext_tiny_frozen": "results/ddr_open_autonomy_specificity300_convnext_tiny/predictions.csv",
    "feature_ensemble": "results/ddr_open_autonomy_specificity300_ensemble/predictions.csv",
    "feature_mlp": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50/predictions.csv",
    "feature_mlp_conservative": "results/ddr_feature_mlp_specificity300_ensemble_h512_d50_conservative/predictions.csv",
    "finetuned_resnet50_e2": "results/ddr_finetuned_autonomy_resnet50_e2/predictions.csv",
    "regenerated_resnet50": "results/ddr_open_autonomy_specificity300_regenerated/predictions.csv",
    "clearance_ranker": "results/clearance_ranker_specificity300_resnet_h768_regenerated/predictions.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score_from_row(row: dict[str, str]) -> float | None:
    for key in ("referable_probability", "score", "risk_score", "referable_score"):
        if key in row and row[key] != "":
            return float(row[key])
    return None


def pred_from_row(row: dict[str, str], score: float | None) -> int | None:
    action = row.get("action", "").upper()
    if action in {"REFER", "AUTONOMOUS_REFERABLE", "DEFER_OR_REFER"}:
        return 1
    if action in {"CLEAR", "AUTONOMOUS_NONREFERABLE", "AUTONOMOUS_NONREFERABLE_CANDIDATE"}:
        return 0
    if score is not None:
        threshold = row.get("threshold")
        if threshold:
            return int(score >= float(threshold))
        return int(score >= 0.5)
    return None


def metric_row(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    test_rows = [r for r in rows if r.get("split") == "test"]
    y: list[int] = []
    pred: list[int] = []
    scores: list[float] = []
    for row in test_rows:
        if "referable_dr" not in row:
            continue
        score = score_from_row(row)
        pp = pred_from_row(row, score)
        if pp is None:
            continue
        y.append(int(row["referable_dr"]))
        pred.append(int(pp))
        scores.append(float(score) if score is not None else float(pp))
    yy = np.asarray(y, dtype=int)
    pp = np.asarray(pred, dtype=int)
    ss = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {
        "model": name,
        "n_test": int(len(yy)),
        "positive": int(yy.sum()) if len(yy) else 0,
        "negative": int((yy == 0).sum()) if len(yy) else 0,
    }
    if len(yy) == 0:
        out["status"] = "NO_ROWS"
        return out
    out.update(
        {
            "accuracy": float(accuracy_score(yy, pp)),
            "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
            "precision": float(precision_score(yy, pp, zero_division=0)),
            "recall_sensitivity": float(recall_score(yy, pp, zero_division=0)),
            "specificity": float(((yy == 0) & (pp == 0)).sum() / max(1, int((yy == 0).sum()))),
            "f1": float(f1_score(yy, pp, zero_division=0)),
            "mcc": float(matthews_corrcoef(yy, pp)),
            "false_negative": int(((yy == 1) & (pp == 0)).sum()),
            "false_positive": int(((yy == 0) & (pp == 1)).sum()),
        }
    )
    if len(set(yy.tolist())) == 2 and len(set(ss.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(yy, ss))
        out["average_precision"] = float(average_precision_score(yy, ss))
    else:
        out["auroc"] = ""
        out["average_precision"] = ""
    out["zero_miss"] = bool(out["false_negative"] == 0)
    out["every_image_90_0"] = bool(out["false_negative"] == 0 and out["specificity"] >= 0.90)
    return out


def fmt_pct(x: Any) -> str:
    if x == "" or x is None:
        return "NA"
    return f"{100 * float(x):.1f}"


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" text-anchor="{anchor}" fill="#1f2933">{safe}</text>'


def write_metric_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = ["auroc", "average_precision", "recall_sensitivity", "specificity", "f1", "mcc"]
    labels = ["AUROC", "AUPRC", "Recall", "Specificity", "F1", "MCC"]
    width = 1040
    cell_w = 104
    cell_h = 34
    left = 260
    top = 70
    height = top + cell_h * len(rows) + 64
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        svg_text(24, 32, "DDR model zoo: held-out test metrics", 18, weight="700"),
    ]
    for j, label in enumerate(labels):
        parts.append(svg_text(left + j * cell_w + cell_w / 2, top - 18, label, 11, anchor="middle", weight="700"))
    for i, row in enumerate(rows):
        y = top + i * cell_h
        parts.append(svg_text(24, y + 22, row["model"], 11))
        for j, key in enumerate(metrics):
            val = row.get(key)
            if val == "" or val is None:
                strength = 0.0
            elif key == "mcc":
                strength = max(0.0, min(1.0, (float(val) + 1) / 2))
            else:
                strength = max(0.0, min(1.0, float(val)))
            # Blue-green scale with enough contrast for print.
            r = int(236 - 170 * strength)
            g = int(246 - 88 * strength)
            b = int(248 - 90 * strength)
            x = left + j * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="rgb({r},{g},{b})" rx="3"/>')
            text = f"{float(val):.2f}" if val != "" and val is not None else "NA"
            parts.append(svg_text(x + cell_w / 2 - 2, y + 20, text, 10, anchor="middle"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_score_distribution(run_name: str, rows: list[dict[str, str]], path: Path) -> None:
    test = [r for r in rows if r.get("split") == "test"]
    pos = [score_from_row(r) for r in test if int(r.get("referable_dr", 0)) == 1]
    neg = [score_from_row(r) for r in test if int(r.get("referable_dr", 0)) == 0]
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    width = 860
    height = 420
    left = 70
    top = 56
    chart_w = 720
    chart_h = 260
    bins = np.linspace(0, 1, 31)
    pos_h, _ = np.histogram(pos, bins=bins)
    neg_h, _ = np.histogram(neg, bins=bins)
    max_h = max(1, int(max(pos_h.max(initial=0), neg_h.max(initial=0))))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        svg_text(24, 32, f"Score distribution: {run_name}", 18, weight="700"),
    ]
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#1f2933"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#1f2933"/>')
    bin_w = chart_w / (len(bins) - 1)
    for i in range(len(bins) - 1):
        x = left + i * bin_w
        nh = chart_h * neg_h[i] / max_h
        ph = chart_h * pos_h[i] / max_h
        parts.append(f'<rect x="{x:.1f}" y="{top + chart_h - nh:.1f}" width="{bin_w * 0.45:.1f}" height="{nh:.1f}" fill="#6aa6b8" opacity="0.75"/>')
        parts.append(f'<rect x="{x + bin_w * 0.45:.1f}" y="{top + chart_h - ph:.1f}" width="{bin_w * 0.45:.1f}" height="{ph:.1f}" fill="#c45252" opacity="0.75"/>')
    parts.append(svg_text(left, top + chart_h + 30, "0.0", 10, anchor="middle"))
    parts.append(svg_text(left + chart_w, top + chart_h + 30, "1.0", 10, anchor="middle"))
    parts.append(svg_text(left + chart_w / 2, height - 28, "Predicted referable-DR score", 13, anchor="middle"))
    parts.append(f'<rect x="{left + 520}" y="{top + 12}" width="14" height="14" fill="#6aa6b8" opacity="0.75"/>')
    parts.append(svg_text(left + 540, top + 24, "Nonreferable", 11))
    parts.append(f'<rect x="{left + 520}" y="{top + 34}" width="14" height="14" fill="#c45252" opacity="0.75"/>')
    parts.append(svg_text(left + 540, top + 46, "Referable", 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, Any]], out: Path) -> None:
    sorted_rows = sorted(rows, key=lambda r: (float(r.get("mcc", -1)), float(r.get("specificity", 0))), reverse=True)
    best_mcc = sorted_rows[0] if sorted_rows else {}
    zero_miss = [r for r in rows if r.get("zero_miss")]
    best_zero_miss = max(zero_miss, key=lambda r: float(r.get("specificity", 0)), default=None)
    every_image_pass = [r for r in rows if r.get("every_image_90_0")]
    lines = [
        "# DDR Model-Zoo Evaluation\n\n",
        "This report converts the current DDR experiments into a paper-ready model-comparison block. It follows the structure of mature biomedical ML figures: multiple model families, balanced metrics, and score-distribution diagnostics rather than a single preferred run.\n\n",
        "## Main Result\n\n",
        f"- Best overall discrimination model by MCC: `{best_mcc.get('model', 'NA')}` with MCC `{float(best_mcc.get('mcc', 0)):.3f}`, AUROC `{float(best_mcc.get('auroc', 0)):.3f}`, AUPRC `{float(best_mcc.get('average_precision', 0)):.3f}`, and specificity `{fmt_pct(best_mcc.get('specificity'))}%` at its logged operating point.\n",
    ]
    if best_zero_miss:
        lines.append(
            f"- Best zero-miss model at the logged operating point: `{best_zero_miss['model']}` with specificity `{fmt_pct(best_zero_miss.get('specificity'))}%` and false negatives `{best_zero_miss.get('false_negative')}`.\n"
        )
    if every_image_pass:
        lines.append("- At least one evaluated every-image model satisfies `false_negative = 0` and `specificity >= 90%`.\n\n")
    else:
        lines.append("- No evaluated every-image model currently satisfies `false_negative = 0` and `specificity >= 90%`.\n\n")
    lines.extend(
        [
            "## Paper Figure Package\n\n",
            "- `results/ddr_model_zoo/model_zoo_metrics.csv`\n",
            "- `results/ddr_model_zoo/model_zoo_metric_heatmap.svg`\n",
            "- `results/ddr_model_zoo/feature_ensemble_score_distribution.svg`\n",
            "- `results/ddr_model_zoo/feature_mlp_score_distribution.svg`\n\n",
            "## Interpretation\n\n",
            "The benchmark shows that high AUROC/AUPRC does not automatically translate into every-image autonomous diagnosis under a zero-miss clearance constraint. This is the central technical gap: the current models rank referable disease well, but their negative-class tail is not clean enough to clear every image autonomously at high specificity.\n\n",
            "The practical paper claim should therefore emphasize risk-controlled selective autonomy now, and use the failed every-image experiments as a falsification analysis. The next model-development step should be an explicitly clearance-trained architecture with hard-positive mining, image-quality gating, and a validation objective that directly maximizes nonreferable clearance subject to zero positive clearances.\n",
        ]
    )
    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path("results/ddr_model_zoo")
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    raw_rows: dict[str, list[dict[str, str]]] = {}
    for name, rel in RUNS.items():
        rows = read_rows(Path(rel))
        raw_rows[name] = rows
        if rows:
            metric_rows.append(metric_row(name, rows))
    write_csv(out_dir / "model_zoo_metrics.csv", metric_rows)
    write_metric_heatmap(metric_rows, out_dir / "model_zoo_metric_heatmap.svg")
    for name in ("feature_ensemble", "feature_mlp"):
        if raw_rows.get(name):
            write_score_distribution(name, raw_rows[name], out_dir / f"{name}_score_distribution.svg")
    write_report(metric_rows, out_dir / "model_zoo_report.md")
    summary = {
        "status": "MODEL_ZOO_REPORT_READY",
        "n_models": len(metric_rows),
        "metrics": str(out_dir / "model_zoo_metrics.csv"),
        "report": str(out_dir / "model_zoo_report.md"),
        "figures": [
            str(out_dir / "model_zoo_metric_heatmap.svg"),
            str(out_dir / "feature_ensemble_score_distribution.svg"),
            str(out_dir / "feature_mlp_score_distribution.svg"),
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
