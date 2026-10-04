"""Search whether an added model score improves three-output autonomy.

This keeps the original score-family search untouched and builds an augmented
matrix with one extra external score file. The gate is the same fail-closed
contract: zero false-negative clearances and >=90% specificity on validation,
then one frozen test readout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from active_retina.autonomy_zero_miss_90_search import metrics, ranknorm, read_model_rows
from active_retina.three_output_autonomous_triage import triage_metrics


def read_extra_rows(path: Path, name: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["image_id"]] = {
                "split": row["split"],
                "image_id": row["image_id"],
                "referable_dr": int(row["referable_dr"]),
                "score": float(row["score"]),
                "source": name,
            }
    return rows


def triage_action(score: float, low: float, high: float) -> str:
    if score <= low:
        return "AUTONOMOUS_NONREFERABLE"
    if score >= high:
        return "AUTONOMOUS_REFERABLE"
    return "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"


def triage_rows(image_ids: list[str], splits: list[str], y: np.ndarray, scores: np.ndarray, low: float, high: float) -> list[dict[str, Any]]:
    rows = []
    for image_id, split, yy, score in zip(image_ids, splits, y, scores):
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "referable_dr": int(yy),
                "score": float(score),
                "low_threshold": float(low),
                "high_threshold": float(high),
                "action": triage_action(float(score), float(low), float(high)),
            }
        )
    return rows


def select_policy_fast(valid_y: np.ndarray, valid_s: np.ndarray, min_spec: float, grid_size: int = 81) -> dict[str, Any]:
    pos_scores = valid_s[valid_y == 1]
    if len(pos_scores) == 0:
        return {"status": "NO_POSITIVES"}
    max_safe_low = float(np.min(pos_scores)) - 1e-12
    quantiles = np.quantile(valid_s, np.linspace(0, 1, grid_size))
    low_grid = sorted({float(x) for x in quantiles if float(x) <= max_safe_low} | {float(np.min(valid_s)) - 1e-9, max_safe_low})
    high_grid = sorted({float(x) for x in quantiles} | {float(np.max(valid_s)) + 1e-9})
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            row = metrics(valid_y, valid_s, low, high)
            if row["false_negative_clearances"] != 0:
                continue
            if row["specificity_among_auto_decisions"] < min_spec:
                continue
            key = (
                row["coverage"],
                row["negative_clearance_rate"],
                row["specificity_among_auto_decisions"],
            )
            if best is None or key > best[0]:
                best = (key, row)
    return best[1] if best else {"status": "NO_VALID_ZERO_MISS_90_POLICY"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-predictions", required=True)
    parser.add_argument("--extra-name", default="extra_signal")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--random-weight-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    models, base_ids, model_rows = read_model_rows()
    extra_rows = read_extra_rows(Path(args.extra_predictions), args.extra_name)
    model_rows[args.extra_name] = extra_rows
    models = sorted(model_rows)
    image_ids = sorted(set(base_ids).intersection(extra_rows))
    splits = [model_rows[models[0]][i]["split"] for i in image_ids]
    y = np.asarray([model_rows[models[0]][i]["referable_dr"] for i in image_ids], dtype=int)
    matrix = np.asarray([[model_rows[m][i]["score"] for m in models] for i in image_ids], dtype=float)

    candidates: dict[str, np.ndarray] = {
        **{m: matrix[:, idx] for idx, m in enumerate(models)},
        "mean_augmented": matrix.mean(axis=1),
        "median_augmented": np.median(matrix, axis=1),
        "max_augmented": matrix.max(axis=1),
        "min_augmented": matrix.min(axis=1),
        "top2_mean_augmented": np.sort(matrix, axis=1)[:, -2:].mean(axis=1),
        "top3_mean_augmented": np.sort(matrix, axis=1)[:, -3:].mean(axis=1),
        "base_mean_plus_extra_mean": 0.5 * matrix[:, [models.index(m) for m in models if m != args.extra_name]].mean(axis=1)
        + 0.5 * matrix[:, models.index(args.extra_name)],
    }
    if args.random_weight_candidates > 0:
        rng = np.random.default_rng(args.seed)
        for idx in range(args.random_weight_candidates):
            weights = rng.dirichlet(np.ones(len(models)))
            scores = matrix @ weights
            nonzero = "_".join(f"{m[:4]}{w:.2f}" for m, w in zip(models, weights) if w >= 0.05)
            candidates[f"random_weight_{idx:04d}_{nonzero}"] = scores
    for name, values in list(candidates.items()):
        candidates[f"{name}_ranknorm"] = ranknorm(values, splits)

    valid_mask = np.asarray([s == "valid" for s in splits])
    test_mask = np.asarray([s == "test" for s in splits])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for name, scores in candidates.items():
        selected = select_policy_fast(y[valid_mask], scores[valid_mask], min_spec=args.min_specificity)
        if "status" in selected:
            rows.append({"candidate": name, "status": selected["status"]})
            continue
        test = metrics(y[test_mask], scores[test_mask], float(selected["low"]), float(selected["high"]))
        status = (
            "GO_ZERO_MISS_90_SPECIFICITY"
            if test["false_negative_clearances"] == 0
            and test["specificity_among_auto_decisions"] >= args.min_specificity
            and test["negative_clearance_rate"] > 0
            else "NO_GO_TEST"
        )
        row = {
            "candidate": name,
            "status": status,
            **{f"valid_{k}": v for k, v in selected.items()},
            **{f"test_{k}": v for k, v in test.items()},
        }
        rows.append(row)
        if best is None or (
            int(status == "GO_ZERO_MISS_90_SPECIFICITY"),
            float(row.get("test_coverage", 0.0)),
            float(row.get("test_negative_clearance_rate", 0.0)),
            float(row.get("test_specificity_among_auto_decisions", 0.0)),
        ) > (
            int(best.get("status") == "GO_ZERO_MISS_90_SPECIFICITY"),
            float(best.get("test_coverage", 0.0)),
            float(best.get("test_negative_clearance_rate", 0.0)),
            float(best.get("test_specificity_among_auto_decisions", 0.0)),
        ):
            best = row

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "search_table.csv", rows)
    best = best or {"status": "NO_CANDIDATES"}
    best_name = str(best.get("candidate", ""))
    best_scores = candidates.get(best_name)
    pred_rows: list[dict[str, Any]] = []
    if best_scores is not None and "valid_low" in best and "valid_high" in best:
        pred_rows = triage_rows(image_ids, splits, y, best_scores, float(best["valid_low"]), float(best["valid_high"]))
        write_csv(out / "best_predictions.csv", pred_rows)
    summary = {
        "status": "GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY" if best.get("status") == "GO_ZERO_MISS_90_SPECIFICITY" else "NO_GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY",
        "extra_name": args.extra_name,
        "extra_predictions": args.extra_predictions,
        "n_images": len(image_ids),
        "models": models,
        "best": best,
        "best_triage_metrics": triage_metrics([r for r in pred_rows if r.get("split") == "test"]) if pred_rows else {},
        "table": str(out / "search_table.csv"),
        "best_predictions": str(out / "best_predictions.csv") if pred_rows else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
