"""Search three-zone selective autonomy policies from saved predictions.

Policy:
    score <= low  -> autonomous nonreferable
    score >= high -> autonomous referable
    otherwise     -> defer

This is the clinically realistic way to pursue high specificity without
pretending uncertain cases are safe autonomous decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            score = row.get("score", row.get("referable_probability"))
            if score is None:
                raise ValueError(f"No score column in {path}")
            rows.append(
                {
                    "split": row["split"],
                    "image_id": row["image_id"],
                    "referable_dr": int(row["referable_dr"]),
                    "score": float(score),
                }
            )
    return rows


def metrics(rows: list[dict[str, Any]], low: float, high: float) -> dict[str, Any]:
    positives = [r for r in rows if r["referable_dr"] == 1]
    negatives = [r for r in rows if r["referable_dr"] == 0]
    cleared_pos = [r for r in positives if r["score"] <= low]
    cleared_neg = [r for r in negatives if r["score"] <= low]
    referred_pos = [r for r in positives if r["score"] >= high]
    referred_neg = [r for r in negatives if r["score"] >= high]
    deferred = [r for r in rows if low < r["score"] < high]
    autonomous = len(rows) - len(deferred)
    return {
        "low": low,
        "high": high,
        "n": len(rows),
        "positive": len(positives),
        "negative": len(negatives),
        "autonomous": autonomous,
        "deferred": len(deferred),
        "coverage": autonomous / len(rows) if rows else 0.0,
        "autonomous_nonreferable": len(cleared_neg) + len(cleared_pos),
        "true_nonreferable_cleared": len(cleared_neg),
        "false_nonreferable_clearance": len(cleared_pos),
        "clearance_sensitivity_safety": 1.0 - (len(cleared_pos) / len(positives) if positives else 0.0),
        "nonreferable_clearance_specificity": len(cleared_neg) / len(negatives) if negatives else 0.0,
        "nonreferable_clearance_ppv": len(cleared_neg) / max(1, len(cleared_neg) + len(cleared_pos)),
        "autonomous_referable": len(referred_pos) + len(referred_neg),
        "true_referable_referred": len(referred_pos),
        "false_referable_referral": len(referred_neg),
        "referable_capture": len(referred_pos) / len(positives) if positives else 0.0,
        "referable_precision": len(referred_pos) / max(1, len(referred_pos) + len(referred_neg)),
    }


def search(valid_rows: list[dict[str, Any]], target_specificity: float, require_zero_false_clearance: bool) -> dict[str, Any]:
    scores = sorted(set(r["score"] for r in valid_rows))
    lows = [min(scores) - 1e-9] + scores
    highs = scores + [max(scores) + 1e-9]
    candidates: list[dict[str, Any]] = []
    for low in lows:
        for high in highs:
            if low >= high:
                continue
            m = metrics(valid_rows, low, high)
            if require_zero_false_clearance and m["false_nonreferable_clearance"] != 0:
                continue
            if m["nonreferable_clearance_specificity"] < target_specificity:
                continue
            candidates.append(m)
    if not candidates:
        return {"status": "NO_VALID_SELECTIVE_POLICY"}
    best = max(
        candidates,
        key=lambda m: (
            m["coverage"],
            m["true_nonreferable_cleared"],
            m["referable_capture"],
            m["referable_precision"],
        ),
    )
    best["status"] = "SELECTIVE_POLICY_FOUND"
    best["n_candidates"] = len(candidates)
    return best


def write_predictions(path: Path, rows: list[dict[str, Any]], low: float, high: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "referable_dr", "score", "selective_action"])
        writer.writeheader()
        for row in rows:
            if row["score"] <= low:
                action = "AUTONOMOUS_NONREFERABLE"
            elif row["score"] >= high:
                action = "AUTONOMOUS_REFERABLE"
            else:
                action = "DEFER"
            writer.writerow({**row, "selective_action": action})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--allow-false-clearance", action="store_true")
    args = parser.parse_args()

    rows = read_predictions(Path(args.predictions))
    valid = [r for r in rows if r["split"] == "valid"]
    test = [r for r in rows if r["split"] == "test"]
    selected = search(valid, args.target_specificity, not args.allow_false_clearance)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if selected.get("status") != "SELECTIVE_POLICY_FOUND":
        summary = {
            "status": "NO_GO_SELECTIVE_AUTONOMY_POLICY",
            "selected": selected,
            "predictions": args.predictions,
        }
    else:
        low = float(selected["low"])
        high = float(selected["high"])
        valid_metrics = metrics(valid, low, high)
        test_metrics = metrics(test, low, high)
        write_predictions(out / "selective_predictions.csv", rows, low, high)
        go = (
            test_metrics["nonreferable_clearance_specificity"] >= args.target_specificity
            and test_metrics["false_nonreferable_clearance"] == 0
            and test_metrics["coverage"] > 0
        )
        summary = {
            "status": "GO_SELECTIVE_AUTONOMY_90_SPECIFICITY" if go else "NO_GO_SELECTIVE_AUTONOMY_90_SPECIFICITY",
            "predictions": args.predictions,
            "target_specificity": args.target_specificity,
            "require_zero_false_nonreferable_clearance": not args.allow_false_clearance,
            "selected_on_validation": selected,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "boundary": "Three-zone selective autonomy: autonomous referable, autonomous nonreferable, or defer. This is not classify-everyone full autonomy.",
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
