"""Build the falsifiable evidence layer for the ActiveRetina-RC manuscript.

This script only uses the public DeepDRiD development prediction files. It does
not evaluate the exposed historical test split.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evaluate_active_retina_rc_policy import evaluate_policy, pair_predictions, read_predictions
from .evaluate_gates import auc_binary, average_precision, brier_score
from .write_fast_paper_pack import (
    BRANCHES,
    candidate_grid,
    confidence_triage_metrics,
    select_confidence_baseline,
    threshold_metrics,
    write_csv,
)


TARGETS = (0.01, 0.02, 0.05, 0.10, 0.15)
MIN_SPECIFICITY_VALUES = (0.30, 0.40, 0.50, 0.60)
DEFERRAL_COSTS = (0, 1, 2, 5, 10)
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_N = 1000
PLOT_W = 1800
PLOT_H = 900


MetricFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def q(values: list[float], lo: float = 0.025, hi: float = 0.975) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(values, [lo, hi]))


def core_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "images": float(metrics["mean_images"]),
        "defer": float(metrics["defer_rate"]),
        "coverage": 1.0 - float(metrics["defer_rate"]),
        "risk": float(metrics.get("missed_referable_dr_risk", metrics.get("missed_referable_risk_decided"))),
        "specificity": float(metrics.get("specificity_among_decided", metrics.get("specificity_decided"))),
        "missed_all_ref": float(metrics.get("missed_referable_dr_fraction_all_referable", metrics.get("missed_referable_fraction_all"))),
    }


def threshold_selector(
    calibration: list[dict[str, Any]],
    score_key: str,
    target: float,
    min_specificity: float,
    images: int,
) -> tuple[MetricFn, dict[str, Any], bool]:
    grid = sorted(set([0.0, 1.0] + candidate_grid([x[score_key] for x in calibration], 10000)))
    candidates = [(t, threshold_metrics(calibration, score_key, t, images)) for t in grid]
    feasible = [
        (t, m)
        for t, m in candidates
        if finite(m["missed_referable_risk_decided"])
        and m["missed_referable_risk_decided"] <= target
        and finite(m["specificity_decided"])
        and m["specificity_decided"] >= min_specificity
    ]
    pool = feasible or [
        (t, m)
        for t, m in candidates
        if finite(m["missed_referable_risk_decided"]) and m["missed_referable_risk_decided"] <= target
    ]
    if not pool:
        pool = candidates
    t, cal = min(pool, key=lambda z: (-safe(z[1]["specificity_decided"]), safe(z[1]["missed_referable_risk_decided"]), z[0]))

    def fn(examples: list[dict[str, Any]], key: str = score_key, threshold: float = t, n_images: int = images) -> dict[str, Any]:
        return threshold_metrics(examples, key, threshold, n_images)

    return fn, {"threshold": t, "calibration": cal}, bool(feasible)


def acquisition_selector(policy: dict[str, Any], target: float) -> tuple[MetricFn | None, dict[str, Any], bool]:
    selected = next((p for p in policy["selected_policies"] if float(p["target_risk"]) == target), None)
    if not selected or selected["status"] != "EMPIRICALLY_SELECTED_UNCERTIFIED":
        return None, {"status": selected["status"] if selected else "MISSING_TARGET"}, False

    def fn(examples: list[dict[str, Any]], p: dict[str, Any] = selected) -> dict[str, Any]:
        return evaluate_policy(examples, p["stop_low"], p["stop_high"], p["final_threshold"])

    return fn, selected, True


def safe(x: Any, default: float = 999.0) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def method_grid(
    calibration: list[dict[str, Any]],
    policy: dict[str, Any],
    target: float,
    min_specificity: float,
) -> dict[str, tuple[MetricFn, dict[str, Any], bool]]:
    methods: dict[str, tuple[MetricFn, dict[str, Any], bool]] = {}
    acq_fn, acq_meta, acq_ok = acquisition_selector(policy, target)
    if acq_fn:
        methods["acquisition"] = (acq_fn, acq_meta, acq_ok)
    triage = select_confidence_baseline(calibration, calibration, target, min_specificity)
    if triage["status"] == "EMPIRICALLY_SELECTED_UNCERTIFIED":
        methods["confidence_triage"] = (
            lambda ex, p=triage: confidence_triage_metrics(ex, p["stop_low"], p["stop_high"]),
            triage,
            True,
        )
    methods["fixed_one"] = threshold_selector(calibration, "first_score", target, min_specificity, 1)
    methods["fixed_two"] = threshold_selector(calibration, "two_view_score", target, min_specificity, 2)
    return methods


def bootstrap_deltas(
    validation: list[dict[str, Any]],
    lhs: MetricFn,
    rhs: MetricFn,
    cost: float,
) -> dict[str, float]:
    clusters = defaultdict(list)
    for ex in validation:
        clusters[ex["patient_id"]].append(ex)
    patients = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reps = []
    for _ in range(BOOTSTRAP_N):
        sample = [eye for p in rng.choice(patients, size=len(patients)) for eye in clusters[p]]
        lm = core_metrics(lhs(sample))
        rm = core_metrics(rhs(sample))
        reps.append(lm["images"] - rm["images"] + cost * (lm["defer"] - rm["defer"]))
    lo, hi = q(reps)
    obs_l = core_metrics(lhs(validation))
    obs_r = core_metrics(rhs(validation))
    obs = obs_l["images"] - obs_r["images"] + cost * (obs_l["defer"] - obs_r["defer"])
    return {"delta_cost": obs, "delta_cost_ci_low": lo, "delta_cost_ci_high": hi}


def view_value_rows(branch: str, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "calibration", "validation"):
        xs = [x for x in examples if x["split"] == split]
        labels = [x["referable_dr"] for x in xs]
        first = [x["first_score"] for x in xs]
        two = [x["two_view_score"] for x in xs]
        if len(set(labels)) < 2:
            continue
        rows.append(
            {
                "branch": branch,
                "split": split,
                "n_eyes": len(xs),
                "first_auroc": auc_binary(labels, first),
                "two_view_auroc": auc_binary(labels, two),
                "auroc_gain": auc_binary(labels, two) - auc_binary(labels, first),
                "first_auprc": average_precision(labels, first),
                "two_view_auprc": average_precision(labels, two),
                "first_brier": brier_score(labels, first),
                "two_view_brier": brier_score(labels, two),
                "brier_delta_two_minus_first": brier_score(labels, two) - brier_score(labels, first),
            }
        )
    return rows


def mechanism_rows(branch: str, validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for grade in sorted({x["grade"] for x in validation}):
        xs = [x for x in validation if x["grade"] == grade]
        if not xs:
            continue
        rows.append(mechanism_summary(branch, f"grade_{grade}", xs))
    ordered = sorted(validation, key=lambda x: abs(x["first_score"] - 0.5))
    qsize = max(1, len(ordered) // 4)
    bins = [
        ("most_uncertain_q1", ordered[:qsize]),
        ("uncertain_q2", ordered[qsize : 2 * qsize]),
        ("certain_q3", ordered[2 * qsize : 3 * qsize]),
        ("most_certain_q4", ordered[3 * qsize :]),
    ]
    for label, xs in bins:
        rows.append(mechanism_summary(branch, label, xs))
    discord = sorted(validation, key=lambda x: abs(x["two_view_score"] - x["first_score"]), reverse=True)
    rows.append(mechanism_summary(branch, "highest_score_discordance_quartile", discord[:qsize]))
    rows.append(mechanism_summary(branch, "lowest_score_discordance_quartile", discord[-qsize:]))
    return rows


def mechanism_summary(branch: str, group: str, xs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [x["referable_dr"] for x in xs]
    first = [x["first_score"] for x in xs]
    two = [x["two_view_score"] for x in xs]
    first_brier = brier_score(labels, first)
    two_brier = brier_score(labels, two)
    helped = sum((y - t) ** 2 < (y - f) ** 2 for y, f, t in zip(labels, first, two))
    hurt = sum((y - t) ** 2 > (y - f) ** 2 for y, f, t in zip(labels, first, two))
    return {
        "branch": branch,
        "group": group,
        "n_eyes": len(xs),
        "referable_rate": sum(labels) / len(labels) if labels else float("nan"),
        "mean_first_score": sum(first) / len(first) if first else float("nan"),
        "mean_two_view_score": sum(two) / len(two) if two else float("nan"),
        "mean_score_delta_two_minus_first": (sum(two) - sum(first)) / len(xs) if xs else float("nan"),
        "first_brier": first_brier,
        "two_view_brier": two_brier,
        "brier_delta_two_minus_first": two_brier - first_brier,
        "fraction_second_view_helped_brier": helped / len(xs) if xs else float("nan"),
        "fraction_second_view_hurt_brier": hurt / len(xs) if xs else float("nan"),
    }


def oracle_rows(branch: str, validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for score_key, label in (("first_score", "fixed_one"), ("two_view_score", "fixed_two")):
        best = max(
            (threshold_metrics(validation, score_key, t, 1 if score_key == "first_score" else 2) for t in candidate_grid([x[score_key] for x in validation], 10000)),
            key=lambda m: (safe(m["specificity_decided"], -1), -safe(m["missed_referable_risk_decided"])),
        )
        rows.append({"branch": branch, "oracle": label, **core_metrics(best)})
    helped = [x for x in validation if (x["referable_dr"] - x["two_view_score"]) ** 2 < (x["referable_dr"] - x["first_score"]) ** 2]
    rows.append(
        {
            "branch": branch,
            "oracle": "hindsight_second_view_brier_help",
            "images": 1 + len(helped) / len(validation),
            "defer": 0.0,
            "coverage": 1.0,
            "risk": float("nan"),
            "specificity": float("nan"),
            "missed_all_ref": float("nan"),
            "fraction_second_view_helped": len(helped) / len(validation),
        }
    )
    return rows


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{key: row.get(key, "") for key in keys} for row in rows]


def svg_frontier(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 760, 460
    margin = 58
    xs = [float(r["images"]) for r in rows if r["split"] == "validation" and finite(r["images"])]
    ys = [float(r["risk"]) for r in rows if r["split"] == "validation" and finite(r["risk"])]
    if not xs or not ys:
        output.write_text("", encoding="utf-8")
        return
    xmin, xmax = min(xs) - 0.05, max(xs) + 0.05
    ymin, ymax = 0.0, max(ys + [0.16]) + 0.02

    def sx(x: float) -> float:
        return margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)

    colors = {"acquisition": "#006d77", "confidence_triage": "#b23a48", "fixed_one": "#555555", "fixed_two": "#7b2cbf"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="14">Mean images per eye</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="14">Validation missed referable DR risk</text>',
    ]
    for row in rows:
        if row["split"] != "validation" or not finite(row["risk"]):
            continue
        color = colors.get(row["method"], "#333333")
        parts.append(f'<circle cx="{sx(float(row["images"])):.1f}" cy="{sy(float(row["risk"])):.1f}" r="4.5" fill="{color}" opacity="0.86"/>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (PLOT_W, PLOT_H), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 34), title, fill="#111111", font=_font(42))
    if subtitle:
        draw.text((72, 88), subtitle, fill="#444444", font=_font(24))
    return img, draw


def _plot_area() -> tuple[int, int, int, int]:
    return 120, 150, PLOT_W - 110, PLOT_H - 120


def _axis(draw: ImageDraw.ImageDraw, xlabel: str, ylabel: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = _plot_area()
    draw.line((left, bottom, right, bottom), fill="#222222", width=3)
    draw.line((left, top, left, bottom), fill="#222222", width=3)
    draw.text(((left + right) // 2 - 130, PLOT_H - 62), xlabel, fill="#222222", font=_font(24))
    draw.text((18, (top + bottom) // 2 - 18), ylabel, fill="#222222", font=_font(24))
    return left, top, right, bottom


def _ticks(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - ymin) / (ymax - ymin) * (bottom - top)

    for t in xticks:
        x = sx(t)
        draw.line((x, bottom, x, bottom + 8), fill="#222222", width=2)
        draw.text((x - 20, bottom + 16), f"{t:.1f}", fill="#333333", font=_font(20))
    for t in yticks:
        y = sy(t)
        draw.line((left - 8, y, left, y), fill="#222222", width=2)
        draw.line((left, y, right, y), fill="#eeeeee", width=1)
        draw.text((left - 78, y - 12), f"{t:.2f}", fill="#333333", font=_font(20))
    return sx, sy


def png_frontier(rows: list[dict[str, Any]], output: Path) -> None:
    img, draw = _canvas(
        "Risk--image frontier from frozen validation predictions",
        "Each point is one calibration-selected operating point; point size encodes automatic coverage.",
    )
    left, top, right, bottom = _axis(draw, "mean images per eye", "miss risk")
    valid = [r for r in rows if r["split"] == "validation" and finite(r["risk"])]
    sx, sy = _ticks(draw, left, top, right, bottom, 0.9, 2.08, 0.0, 0.18, [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], [0.00, 0.05, 0.10, 0.15])
    colors = {"acquisition": "#006d77", "confidence_triage": "#b23a48", "fixed_one": "#555555", "fixed_two": "#7b2cbf"}
    offsets = {"suffix2_initial": -5, "suffix1_initial": 5}
    for row in valid:
        x = sx(float(row["images"]))
        y = sy(float(row["risk"])) + offsets.get(row["branch"], 0)
        radius = 7 + 18 * float(row["coverage"])
        fill = colors.get(row["method"], "#333333")
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="white", width=3)
    draw.line((left, sy(0.05), right, sy(0.05)), fill="#d00000", width=3)
    draw.text((right - 210, sy(0.05) - 34), "5% risk reference", fill="#d00000", font=_font(22))
    legend_x, legend_y = right - 420, top + 18
    for i, (name, color) in enumerate(colors.items()):
        y = legend_y + i * 36
        draw.ellipse((legend_x, y, legend_x + 24, y + 24), fill=color)
        draw.text((legend_x + 38, y - 2), name.replace("_", " "), fill="#222222", font=_font(22))
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def png_view_value(rows: list[dict[str, Any]], output: Path) -> None:
    img, draw = _canvas(
        "Paired-view value on validation",
        "Positive AUROC gain and negative Brier delta indicate the second view adds useful evidence.",
    )
    left, top, right, bottom = _axis(draw, "branch", "change")
    vals = [r for r in rows if r["split"] == "validation"]
    baseline_y = (top + bottom) // 2
    draw.line((left, baseline_y, right, baseline_y), fill="#222222", width=2)
    yscale = 260
    bar_w = 95
    x0 = left + 250
    for i, row in enumerate(vals):
        center = x0 + i * 380
        gain = float(row["auroc_gain"])
        brier_improve = -float(row["brier_delta_two_minus_first"])
        for j, (label, value, color) in enumerate((("AUROC gain", gain, "#006d77"), ("Brier improvement", brier_improve, "#e29578"))):
            x = center + (j - 0.5) * 130
            y = baseline_y - value * yscale * 4
            draw.rectangle((x - bar_w / 2, min(y, baseline_y), x + bar_w / 2, max(y, baseline_y)), fill=color)
            draw.text((x - 50, min(y, baseline_y) - 34), f"{value:.3f}", fill="#222222", font=_font(21))
        draw.text((center - 95, bottom + 22), row["branch"].replace("_", " "), fill="#222222", font=_font(22))
    draw.text((right - 360, top + 24), "AUROC gain", fill="#006d77", font=_font(24))
    draw.text((right - 360, top + 58), "Brier improvement", fill="#e29578", font=_font(24))
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def png_mechanism(rows: list[dict[str, Any]], output: Path) -> None:
    wanted = [
        ("suffix2_initial", "grade_2"),
        ("suffix2_initial", "highest_score_discordance_quartile"),
        ("suffix2_initial", "most_uncertain_q1"),
        ("suffix1_initial", "grade_1"),
        ("suffix1_initial", "highest_score_discordance_quartile"),
        ("suffix1_initial", "grade_2"),
    ]
    lookup = {(r["branch"], r["group"]): r for r in rows}
    vals = [lookup[k] for k in wanted if k in lookup]
    img, draw = _canvas(
        "Where the second view changes prediction quality",
        "Bars show Brier delta (two-view minus initial); negative values favor acquiring the paired view.",
    )
    left, top, right, bottom = _axis(draw, "diagnostic subgroup", "Brier delta")
    baseline_y = top + (bottom - top) * 0.48
    draw.line((left, baseline_y, right, baseline_y), fill="#222222", width=2)
    scale = 950
    bar_w = 100
    spacing = (right - left - 160) / max(1, len(vals))
    for i, row in enumerate(vals):
        value = float(row["brier_delta_two_minus_first"])
        x = left + 110 + i * spacing
        y = baseline_y + value * scale
        color = "#006d77" if value < 0 else "#b23a48"
        draw.rectangle((x - bar_w / 2, min(y, baseline_y), x + bar_w / 2, max(y, baseline_y)), fill=color)
        branch = "suffix 2" if row["branch"] == "suffix2_initial" else "suffix 1"
        group = {
            "grade_2": "grade 2",
            "grade_1": "grade 1",
            "highest_score_discordance_quartile": "high discordance",
            "most_uncertain_q1": "most uncertain",
        }.get(row["group"], row["group"].replace("_", " "))
        label = f"{branch}\n{group}"
        draw.multiline_text((x - 94, bottom + 16), label, fill="#222222", font=_font(21), spacing=3, align="center")
        text_y = min(y, baseline_y) - 34 if value < 0 else max(y, baseline_y) + 8
        draw.text((x - 42, text_y), f"{value:.3f}", fill="#222222", font=_font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def png_cost_sensitivity(rows: list[dict[str, Any]], output: Path) -> None:
    img, draw = _canvas(
        "Acquisition-minus-triage operational cost",
        "Negative values favor acquisition; intervals are paired patient-cluster bootstrap percentiles.",
    )
    left, top, right, bottom = _axis(draw, "deferral cost (image-equivalents)", "delta cost")
    chosen = [r for r in rows if float(r["min_specificity"]) == 0.5 and float(r["target_risk"]) in {0.01, 0.10}]
    y_values = []
    for r in chosen:
        y_values.extend([float(r["delta_cost"]), float(r["delta_cost_ci_low"]), float(r["delta_cost_ci_high"])])
    ymin = min(-0.45, min(y_values) - 0.15)
    ymax = max(1.65, max(y_values) + 0.20)
    sx, sy = _ticks(draw, left, top, right, bottom, 0, 10, ymin, ymax, [0, 2, 5, 10], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    draw.line((left, sy(0), right, sy(0)), fill="#111111", width=2)
    styles = {
        ("suffix2_initial", 0.01): ("#006d77", "suffix2, risk 1%"),
        ("suffix1_initial", 0.01): ("#83c5be", "suffix1, risk 1%"),
        ("suffix2_initial", 0.10): ("#b23a48", "suffix2, risk 10%"),
        ("suffix1_initial", 0.10): ("#e29578", "suffix1, risk 10%"),
    }
    for key, (color, label) in styles.items():
        branch, target = key
        pts = sorted([r for r in chosen if r["branch"] == branch and float(r["target_risk"]) == target], key=lambda r: float(r["deferral_cost"]))
        coords = [(sx(float(r["deferral_cost"])), sy(float(r["delta_cost"]))) for r in pts]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=5)
        for row, (x, y) in zip(pts, coords):
            lo = sy(float(row["delta_cost_ci_low"]))
            hi = sy(float(row["delta_cost_ci_high"]))
            draw.line((x, lo, x, hi), fill=color, width=2)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    legend_x, legend_y = right - 360, top + 18
    for i, (_key, (color, label)) in enumerate(styles.items()):
        y = legend_y + i * 34
        draw.line((legend_x, y + 12, legend_x + 34, y + 12), fill=color, width=5)
        draw.text((legend_x + 46, y), label, fill="#222222", font=_font(20))
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    frontier_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    mech_rows: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []

    for branch in BRANCHES:
        raw = read_predictions(branch["predictions"])
        if any(r["split"] not in {"train", "calibration", "validation"} for r in raw):
            raise SystemExit("Exposed or unknown split found in development prediction file.")
        patients = {s: {r["patient_id"] for r in raw if r["split"] == s} for s in ("train", "calibration", "validation")}
        if patients["train"] & patients["calibration"] or patients["train"] & patients["validation"] or patients["calibration"] & patients["validation"]:
            raise SystemExit("Patient split leakage detected.")
        pairs = list(pair_predictions(raw, branch["first_view"], branch["second_view"]).values())
        calibration = [x for x in pairs if x["split"] == "calibration"]
        validation = [x for x in pairs if x["split"] == "validation"]
        policy = json.loads(branch["policy"].read_text(encoding="utf-8"))
        source_audit.append(
            {
                "branch": branch["name"],
                "prediction_file": str(branch["predictions"]),
                "sha256": hashlib.sha256(branch["predictions"].read_bytes()).hexdigest(),
                "patients": {s: len(v) for s, v in patients.items()},
                "calibration_eyes": len(calibration),
                "validation_eyes": len(validation),
            }
        )
        view_rows.extend(view_value_rows(branch["name"], pairs))
        mech_rows.extend(mechanism_rows(branch["name"], validation))
        oracle.extend(oracle_rows(branch["name"], validation))
        for target in TARGETS:
            for min_spec in MIN_SPECIFICITY_VALUES:
                methods = method_grid(calibration, policy, target, min_spec)
                valid_fns: dict[str, MetricFn] = {}
                for method, (fn, meta, feasible) in methods.items():
                    valid_fns[method] = fn
                    for split_name, examples in (("calibration", calibration), ("validation", validation)):
                        metrics = core_metrics(fn(examples))
                        frontier_rows.append(
                            {
                                "branch": branch["name"],
                                "target_risk": target,
                                "min_specificity": min_spec,
                                "method": method,
                                "split": split_name,
                                "calibration_feasible": feasible,
                                **metrics,
                            }
                        )
                if "acquisition" in valid_fns and "confidence_triage" in valid_fns:
                    for cost in DEFERRAL_COSTS:
                        cost_rows.append(
                            {
                                "branch": branch["name"],
                                "target_risk": target,
                                "min_specificity": min_spec,
                                "deferral_cost": cost,
                                **bootstrap_deltas(validation, valid_fns["acquisition"], valid_fns["confidence_triage"], cost),
                            }
                        )

    write_csv(Path("results/flagship_frontier.csv"), normalize_rows(frontier_rows))
    write_csv(Path("results/flagship_cost_frontier.csv"), normalize_rows(cost_rows))
    write_csv(Path("results/flagship_view_value.csv"), normalize_rows(view_rows))
    write_csv(Path("results/flagship_mechanism.csv"), normalize_rows(mech_rows))
    write_csv(Path("results/flagship_oracle_bounds.csv"), normalize_rows(oracle))
    svg_frontier(frontier_rows, Path("plots/flagship_risk_image_frontier.svg"))
    png_frontier(frontier_rows, Path("plots/flagship_risk_image_frontier.png"))
    png_view_value(view_rows, Path("plots/flagship_paired_view_value.png"))
    png_mechanism(mech_rows, Path("plots/flagship_mechanism_brier.png"))
    png_cost_sensitivity(cost_rows, Path("plots/flagship_cost_sensitivity.png"))
    Path("results/flagship_decision.json").write_text(
        json.dumps(
            {
                "status": "GO_DEEPDRID_ARXIV_BENCHMARK_CONDITIONAL_IEEE",
                "arxiv_status": "GO_AS_TRANSPARENT_REPRODUCIBLE_BENCHMARK_PREPRINT",
                "ieee_status": "CONDITIONAL_GO_REQUIRES_STRONGER_DEEPDRID_ONLY_RISK_CONTROL",
                "core_hypothesis": "Acquisition must improve the risk-coverage-cost frontier over confidence triage on independent patient-disjoint data.",
                "source_audit": source_audit,
                "go_claim": [
                    "The paper is a GO as a reproducible DeepDRiD-only benchmark and methods preprint: it has real paired-view predictions, patient-disjoint development evaluation, risk-coverage-cost baselines, bootstrap uncertainty, mapping sensitivity, mechanism diagnostics, generated figures, and exact calibration-set risk upper bounds.",
                    "The paper is not a GO for an unqualified clinical or IEEE-journal superiority claim until the DeepDRiD-only ML stack can provide stronger prespecified finite-sample risk control and calibration.",
                ],
                "conditional_ieee_blockers": [
                    "DeepDRiD suffix-to-anatomy mapping is not officially resolved in reviewed public metadata.",
                    "Current validation has 39 patients; uncertainty is too wide for an unqualified IEEE safety claim.",
                    "Exact calibration-set risk bounds do not certify a 5% missed-referable-DR target with the present sample.",
                    "The conservative acquisition policy trades lower observed missed referable DR for lower coverage, higher imaging, and higher deferral versus confidence triage.",
                ],
                "completed_artifacts": [
                    "results/flagship_frontier.csv",
                    "results/flagship_cost_frontier.csv",
                    "results/flagship_mechanism.csv",
                    "results/flagship_oracle_bounds.csv",
                    "plots/flagship_risk_image_frontier.svg",
                    "plots/flagship_risk_image_frontier.png",
                    "plots/flagship_paired_view_value.png",
                    "plots/flagship_mechanism_brier.png",
                    "plots/flagship_cost_sensitivity.png",
                    "results/formal_risk_control_summary.json",
                    "results/formal_risk_control_selected.csv",
                    "results/formal_risk_control_candidates.csv",
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
