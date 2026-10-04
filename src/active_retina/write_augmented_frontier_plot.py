"""Write an SVG frontier plot from real augmented autonomy search results."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


RUNS = {
    "baseline": Path("results/three_output_autonomous_triage/summary.json"),
    "augmented_max_coverage": Path("results/autonomy_extra_signal_clementp_efficientnet_b5_random2000_fast/summary.json"),
    "augmented_conservative": Path("results/autonomy_extra_signal_clementp_efficientnet_b5_random600_spec925/summary.json"),
}
TABLES = [
    Path("results/autonomy_extra_signal_clementp_efficientnet_b5/search_table.csv"),
    Path("results/autonomy_extra_signal_clementp_efficientnet_b5_random600/search_table.csv"),
    Path("results/autonomy_extra_signal_clementp_efficientnet_b5_random600_spec925/search_table.csv"),
]
OUT = Path("results/autonomy_extra_signal_clementp_efficientnet_b5/frontier.svg")


def read_points() -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    for table in TABLES:
        if not table.exists():
            continue
        with table.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    coverage = float(row.get("test_coverage", "nan"))
                    specificity = float(row.get("test_specificity_among_auto_decisions", "nan"))
                    misses = int(float(row.get("test_false_negative_clearances", "999")))
                except ValueError:
                    continue
                if coverage == coverage and specificity == specificity:
                    points.append(
                        {
                            "candidate": row.get("candidate", ""),
                            "coverage": coverage,
                            "specificity": specificity,
                            "misses": misses,
                        }
                    )
    return points


def load_named_points() -> dict[str, dict[str, float]]:
    baseline = json.loads(RUNS["baseline"].read_text(encoding="utf-8"))["test_triage_metrics"]
    named = {
        "baseline": {
            "coverage": float(baseline["diagnostic_decision_rate"]),
            "specificity": float(baseline["specificity_among_diagnostic_decisions"]),
        }
    }
    for name in ("augmented_max_coverage", "augmented_conservative"):
        summary = json.loads(RUNS[name].read_text(encoding="utf-8"))
        best = summary["best_triage_metrics"]
        named[name] = {
            "coverage": float(best["diagnostic_decision_rate"]),
            "specificity": float(best["specificity_among_diagnostic_decisions"]),
        }
    return named


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    points = read_points()
    named = load_named_points()
    width, height = 920, 560
    margin_left, margin_right, margin_top, margin_bottom = 82, 36, 48, 72
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_of(coverage: float) -> float:
        return margin_left + coverage * plot_w

    def y_of(specificity: float) -> float:
        return margin_top + (1.0 - specificity) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="82" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700">ActiveRetina Augmented Autonomy Frontier</text>',
        '<line x1="82" y1="48" x2="82" y2="488" stroke="#222" stroke-width="1.4"/>',
        '<line x1="82" y1="488" x2="884" y2="488" stroke="#222" stroke-width="1.4"/>',
    ]
    for tick in range(0, 101, 10):
        x = x_of(tick / 100)
        parts.append(f'<line x1="{x:.1f}" y1="488" x2="{x:.1f}" y2="494" stroke="#222"/>')
        parts.append(f'<text x="{x:.1f}" y="512" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">{tick}%</text>')
    for tick in range(0, 101, 10):
        y = y_of(tick / 100)
        parts.append(f'<line x1="76" y1="{y:.1f}" x2="82" y2="{y:.1f}" stroke="#222"/>')
        parts.append(f'<text x="68" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11">{tick}%</text>')
    parts.append('<text x="483" y="548" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Diagnostic decision rate</text>')
    parts.append('<text x="20" y="268" transform="rotate(-90 20 268)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Specificity among diagnostic decisions</text>')
    parts.append(f'<line x1="82" y1="{y_of(0.90):.1f}" x2="884" y2="{y_of(0.90):.1f}" stroke="#c0392b" stroke-width="1.2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="888" y="{y_of(0.90)+4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#c0392b">90%</text>')

    for point in points:
        color = "#2f80ed" if int(point["misses"]) == 0 else "#b8c2cc"
        opacity = "0.62" if int(point["misses"]) == 0 else "0.28"
        parts.append(
            f'<circle cx="{x_of(float(point["coverage"])):.1f}" cy="{y_of(float(point["specificity"])):.1f}" r="3.2" fill="{color}" fill-opacity="{opacity}">'
            f'<title>{html.escape(str(point["candidate"]))}: coverage={100*float(point["coverage"]):.1f}%, specificity={100*float(point["specificity"]):.1f}%, misses={point["misses"]}</title></circle>'
        )

    labels = {
        "baseline": ("Baseline", "#111827", -8, -12),
        "augmented_max_coverage": ("Augmented max coverage", "#0b7a3b", 8, -12),
        "augmented_conservative": ("Augmented conservative", "#7c3aed", 8, 18),
    }
    for key, values in named.items():
        label, color, dx, dy = labels[key]
        x = x_of(values["coverage"])
        y = y_of(values["specificity"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        parts.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="{color}">{label}</text>')
    parts.append('<text x="82" y="535" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">Blue points are zero-false-clearance candidate policies; gray points had at least one false nonreferable clearance.</text>')
    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
