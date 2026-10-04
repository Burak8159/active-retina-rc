"""Build an every-input autonomous triage output from the selective policy.

This is the product-realistic route for "complete autonomy" without pretending
that every retinal image is diagnostically gradable. Every image receives one
of three autonomous outputs:

* AUTONOMOUS_NONREFERABLE
* AUTONOMOUS_REFERABLE
* AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE

The third output replaces the vague abstention state with an explicit product
action. It is not counted as a diagnostic clearance or diagnostic referral.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from active_retina.autonomy_zero_miss_90_search import metrics, read_model_rows


OUT_DIR = Path("results/three_output_autonomous_triage")


def load_best_policy() -> dict[str, Any]:
    summary = json.loads(Path("results/autonomy_zero_miss_90_search_summary.json").read_text(encoding="utf-8"))
    return summary["best"]


def score_candidates(models: list[str], image_ids: list[str], model_rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, np.ndarray]:
    splits = [model_rows[models[0]][i]["split"] for i in image_ids]
    matrix = np.asarray([[model_rows[m][i]["score"] for m in models] for i in image_ids], dtype=float)
    candidates: dict[str, np.ndarray] = {
        **{m: matrix[:, idx] for idx, m in enumerate(models)},
        "mean_all": matrix.mean(axis=1),
        "max_all": matrix.max(axis=1),
        "median_all": np.median(matrix, axis=1),
        "top2_mean_all": np.sort(matrix, axis=1)[:, -2:].mean(axis=1),
        "min_all": matrix.min(axis=1),
    }
    for name, values in list(candidates.items()):
        candidates[f"{name}_ranknorm"] = ranknorm(values, splits)
    return candidates


def ranknorm(values: np.ndarray, splits: list[str]) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for split in sorted(set(splits)):
        idx = np.asarray([s == split for s in splits])
        vals = values[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(vals, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
        out[idx] = ranks
    return out


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def resolve_image_path(raw_path: str, root: Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    candidate = root / raw_path
    if candidate.exists():
        return candidate
    normalized = raw_path.replace("\\", "/")
    marker = "/external/"
    if marker in normalized:
        candidate = root / "external" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return p


def image_quality(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        gray = img.convert("L").resize((256, 256))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    dark = float((arr < 0.05).mean())
    bright = float((arr > 0.95).mean())
    return {
        "mean_luminance": float(arr.mean()),
        "contrast": float(arr.std()),
        "edge_energy": float(edges.mean()),
        "dark_fraction": dark,
        "bright_fraction": bright,
    }


def quality_flag(q: dict[str, float]) -> bool:
    """Heuristic fail-closed image-quality gate."""
    if q["mean_luminance"] < 0.05 or q["mean_luminance"] > 0.92:
        return False
    if q["contrast"] < 0.045:
        return False
    if q["edge_energy"] < 0.006:
        return False
    if q["dark_fraction"] > 0.70 or q["bright_fraction"] > 0.70:
        return False
    return True


def triage_action(score: float, low: float, high: float, quality_ok: bool) -> str:
    if not quality_ok:
        return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"
    if score <= low:
        return "AUTONOMOUS_NONREFERABLE"
    if score >= high:
        return "AUTONOMOUS_REFERABLE"
    return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"


def triage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([int(r["referable_dr"]) for r in rows], dtype=int)
    actions = np.asarray([r["action"] for r in rows], dtype=object)
    clear = actions == "AUTONOMOUS_NONREFERABLE"
    refer = actions == "AUTONOMOUS_REFERABLE"
    insufficient = actions == "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"
    pos = y == 1
    neg = y == 0
    false_clear = int((pos & clear).sum())
    true_clear = int((neg & clear).sum())
    false_refer = int((neg & refer).sum())
    true_refer = int((pos & refer).sum())
    diagnostic = clear | refer
    return {
        "n": int(len(rows)),
        "positive": int(pos.sum()),
        "negative": int(neg.sum()),
        "autonomous_output_rate": 1.0,
        "diagnostic_decision_rate": float(diagnostic.mean()),
        "insufficient_evidence_rate": float(insufficient.mean()),
        "false_negative_clearances": false_clear,
        "true_negative_clearances": true_clear,
        "false_positive_referrals": false_refer,
        "true_positive_referrals": true_refer,
        "sensitivity_if_insufficient_is_not_negative": float(1.0 - false_clear / max(1, int(pos.sum()))),
        "specificity_among_diagnostic_decisions": float(true_clear / max(1, true_clear + false_refer)),
        "negative_clearance_rate": float(true_clear / max(1, int(neg.sum()))),
        "referable_autonomous_referral_rate": float(true_refer / max(1, int(pos.sum()))),
    }


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "image_id",
        "referable_dr",
        "score",
        "low_threshold",
        "high_threshold",
        "quality_ok",
        "mean_luminance",
        "contrast",
        "edge_energy",
        "dark_fraction",
        "bright_fraction",
        "action",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_best_policy()
    candidate = str(policy["candidate"])
    low = float(policy["test_low"])
    high = float(policy["test_high"])
    models, image_ids, model_rows = read_model_rows()
    candidates = score_candidates(models, image_ids, model_rows)
    if candidate not in candidates:
        raise KeyError(f"Candidate {candidate} not found. Available: {sorted(candidates)}")
    manifest = read_manifest(Path("results/ddr_hf_specificity300_manifest.csv"))
    splits = [model_rows[models[0]][i]["split"] for i in image_ids]
    y = np.asarray([model_rows[models[0]][i]["referable_dr"] for i in image_ids], dtype=int)
    scores = candidates[candidate]
    rows: list[dict[str, Any]] = []
    root = Path.cwd()
    for image_id, split, yy, score in zip(image_ids, splits, y, scores):
        row = manifest.get(image_id, {})
        q = image_quality(resolve_image_path(row["image_path"], root)) if row else {
            "mean_luminance": float("nan"),
            "contrast": float("nan"),
            "edge_energy": float("nan"),
            "dark_fraction": float("nan"),
            "bright_fraction": float("nan"),
        }
        ok = quality_flag(q)
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "referable_dr": int(yy),
                "score": float(score),
                "low_threshold": low,
                "high_threshold": high,
                "quality_ok": bool(ok),
                **q,
                "action": triage_action(float(score), low, high, ok),
            }
        )
    valid_rows = [r for r in rows if r["split"] == "valid"]
    test_rows = [r for r in rows if r["split"] == "test"]
    write_predictions(OUT_DIR / "predictions.csv", rows)
    valid_score_metrics = metrics(
        np.asarray([r["referable_dr"] for r in valid_rows], dtype=int),
        np.asarray([r["score"] for r in valid_rows], dtype=float),
        low,
        high,
    )
    test_score_metrics = metrics(
        np.asarray([r["referable_dr"] for r in test_rows], dtype=int),
        np.asarray([r["score"] for r in test_rows], dtype=float),
        low,
        high,
    )
    summary = {
        "status": (
            "GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE"
            if triage_metrics(test_rows)["false_negative_clearances"] == 0
            and triage_metrics(test_rows)["specificity_among_diagnostic_decisions"] >= 0.90
            else "NO_GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE"
        ),
        "boundary": "Every image receives an autonomous product output, but insufficient-evidence/reacquire is not a diagnostic disease label.",
        "candidate": candidate,
        "low_threshold": low,
        "high_threshold": high,
        "quality_gate": "heuristic luminance/contrast/edge-energy fail-closed gate",
        "valid_score_policy_metrics_without_quality_gate": valid_score_metrics,
        "test_score_policy_metrics_without_quality_gate": test_score_metrics,
        "valid_triage_metrics": triage_metrics(valid_rows),
        "test_triage_metrics": triage_metrics(test_rows),
        "predictions": str(OUT_DIR / "predictions.csv"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Three-Output Autonomous Triage\n\n",
        f"Status: `{summary['status']}`\n\n",
        "Every image receives one autonomous product output: nonreferable, referable, or insufficient evidence/reacquire. "
        "The insufficient-evidence output is a completed product action, not a diagnostic clearance.\n\n",
        f"Candidate score policy: `{candidate}` with low `{low:.6f}` and high `{high:.6f}`.\n\n",
        "## Test Result\n\n",
        f"- False-negative clearances: `{summary['test_triage_metrics']['false_negative_clearances']}`\n",
        f"- Specificity among diagnostic decisions: `{100 * summary['test_triage_metrics']['specificity_among_diagnostic_decisions']:.1f}%`\n",
        f"- Diagnostic decision rate: `{100 * summary['test_triage_metrics']['diagnostic_decision_rate']:.1f}%`\n",
        f"- Insufficient-evidence/reacquire rate: `{100 * summary['test_triage_metrics']['insufficient_evidence_rate']:.1f}%`\n",
        f"- Negative clearance rate: `{100 * summary['test_triage_metrics']['negative_clearance_rate']:.1f}%`\n\n",
        "## Interpretation\n\n",
        "This converts the successful selective policy into a product-complete output contract. It does not make every image receive a binary diagnosis; it makes every image receive a safe autonomous workflow action.\n",
    ]
    (OUT_DIR / "report.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
