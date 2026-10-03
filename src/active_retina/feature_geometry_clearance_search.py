"""Search conservative feature-geometry clearance policies.

This complements learned classifiers with a fail-closed rule: an image may be
autonomously cleared as nonreferable only if it is geometrically unlike the
referable training set and its nearest neighbors are predominantly
nonreferable. Thresholds are selected on validation only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def load_split(rows: list[dict[str, Any]], features: dict[str, np.ndarray], split: str):
    selected = [r for r in rows if r["source_split"] == f"ddr_{split}" and r["image_id"] in features]
    x = np.vstack([features[r["image_id"]] for r in selected]).astype("float32")
    y = np.asarray([r["referable_dr"] for r in selected], dtype=np.int64)
    grade = np.asarray([r["grade"] for r in selected], dtype=np.int64)
    ids = [r["image_id"] for r in selected]
    return x, y, grade, ids


def standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_z = (train - mean) / std
    other_z = [(x - mean) / std for x in others]
    return train_z.astype("float32"), *(x.astype("float32") for x in other_z)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return (x / norm).astype("float32")


def knn_features(
    query: np.ndarray,
    train: np.ndarray,
    train_y: np.ndarray,
    train_grade: np.ndarray,
    k_values: list[int],
    chunk_size: int,
) -> dict[str, np.ndarray]:
    max_k = max(k_values)
    n = query.shape[0]
    min_pos_dist = np.full(n, np.inf, dtype="float32")
    min_g2_dist = np.full(n, np.inf, dtype="float32")
    pos_idx = train_y == 1
    g2_idx = train_grade == 2
    out: dict[str, np.ndarray] = {}
    for k in k_values:
        out[f"knn_pos_frac_{k}"] = np.zeros(n, dtype="float32")
        out[f"knn_grade_mean_{k}"] = np.zeros(n, dtype="float32")
        out[f"knn_neg_grade0_frac_{k}"] = np.zeros(n, dtype="float32")

    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        q = query[start:stop]
        # Squared Euclidean distance. The features are standardized and
        # optionally L2-normalized before this function.
        d = ((q[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
        if pos_idx.any():
            min_pos_dist[start:stop] = d[:, pos_idx].min(axis=1)
        if g2_idx.any():
            min_g2_dist[start:stop] = d[:, g2_idx].min(axis=1)
        idx = np.argpartition(d, kth=max_k - 1, axis=1)[:, :max_k]
        row = np.arange(idx.shape[0])[:, None]
        local_d = d[row, idx]
        order = np.argsort(local_d, axis=1)
        idx = idx[row, order]
        for k in k_values:
            nn = idx[:, :k]
            nn_y = train_y[nn]
            nn_g = train_grade[nn]
            out[f"knn_pos_frac_{k}"][start:stop] = nn_y.mean(axis=1)
            out[f"knn_grade_mean_{k}"][start:stop] = nn_g.mean(axis=1)
            out[f"knn_neg_grade0_frac_{k}"][start:stop] = ((nn_y == 0) & (nn_g == 0)).mean(axis=1)
    out["min_pos_dist"] = min_pos_dist
    out["min_grade2_dist"] = min_g2_dist
    return out


def metrics(y: np.ndarray, clear: np.ndarray) -> dict[str, Any]:
    pos = y == 1
    neg = y == 0
    false_clear = int((pos & clear).sum())
    true_clear = int((neg & clear).sum())
    return {
        "n": int(len(y)),
        "positive": int(pos.sum()),
        "negative": int(neg.sum()),
        "true_nonreferable_cleared": true_clear,
        "false_nonreferable_clearance": false_clear,
        "nonreferable_clearance_specificity": float(true_clear / max(1, int(neg.sum()))),
        "clearance_coverage": float(clear.mean()),
        "zero_miss": false_clear == 0,
    }


def apply_rule(stats: dict[str, np.ndarray], rule: dict[str, Any]) -> np.ndarray:
    clear = np.ones_like(stats["min_pos_dist"], dtype=bool)
    clear &= stats["min_pos_dist"] >= rule["min_pos_dist_ge"]
    clear &= stats["min_grade2_dist"] >= rule["min_grade2_dist_ge"]
    for key, value in rule["max_pos_frac"].items():
        clear &= stats[key] <= value
    for key, value in rule["min_grade0_frac"].items():
        clear &= stats[key] >= value
    return clear


def search_rules(valid_y: np.ndarray, valid_stats: dict[str, np.ndarray], target_specificity: float) -> dict[str, Any]:
    pos_dist_q = np.quantile(valid_stats["min_pos_dist"], np.linspace(0.0, 0.95, 40))
    g2_dist_q = np.quantile(valid_stats["min_grade2_dist"], np.linspace(0.0, 0.95, 30))
    k_keys = sorted(k for k in valid_stats if k.startswith("knn_pos_frac_"))
    grade0_keys = sorted(k for k in valid_stats if k.startswith("knn_neg_grade0_frac_"))
    max_pos_fracs = [0.0, 0.05, 0.10, 0.20, 0.34]
    min_g0_fracs = [0.0, 0.25, 0.50, 0.67, 0.80]

    best: dict[str, Any] | None = None
    checked = 0
    for min_pos in pos_dist_q:
        for min_g2 in g2_dist_q:
            for k_key in k_keys:
                for g0_key in grade0_keys:
                    for max_pf in max_pos_fracs:
                        for min_g0 in min_g0_fracs:
                            rule = {
                                "min_pos_dist_ge": float(min_pos),
                                "min_grade2_dist_ge": float(min_g2),
                                "max_pos_frac": {k_key: float(max_pf)},
                                "min_grade0_frac": {g0_key: float(min_g0)},
                            }
                            checked += 1
                            clear = apply_rule(valid_stats, rule)
                            m = metrics(valid_y, clear)
                            if not m["zero_miss"]:
                                continue
                            if best is None or (
                                m["nonreferable_clearance_specificity"],
                                m["clearance_coverage"],
                            ) > (
                                best["valid_metrics"]["nonreferable_clearance_specificity"],
                                best["valid_metrics"]["clearance_coverage"],
                            ):
                                best = {"rule": rule, "valid_metrics": m}
                            if m["nonreferable_clearance_specificity"] >= target_specificity:
                                best = {"rule": rule, "valid_metrics": m, "early_target_hit": True, "rules_checked": checked}
                                return best
    return best or {"status": "NO_ZERO_MISS_RULE", "rules_checked": checked}


def write_predictions(path: Path, split: str, ids: list[str], y: np.ndarray, grade: np.ndarray, clear: np.ndarray) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "grade", "referable_dr", "action"])
        if not exists:
            writer.writeheader()
        for image_id, gg, yy, cc in zip(ids, grade, y, clear):
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "grade": int(gg),
                    "referable_dr": int(yy),
                    "action": "AUTONOMOUS_NONREFERABLE" if cc else "DEFER_OR_REFER",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--k-values", default="3,5,9,15,25,51")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--l2-normalize", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    raw = np.load(args.feature_cache, allow_pickle=False)
    features = {k: raw[k].astype("float32") for k in raw.files}
    x_train, y_train, g_train, _ = load_split(rows, features, "train")
    x_valid, y_valid, g_valid, valid_ids = load_split(rows, features, "valid")
    x_test, y_test, g_test, test_ids = load_split(rows, features, "test")
    x_train, x_valid, x_test = standardize(x_train, x_valid, x_test)
    if args.l2_normalize:
        x_train, x_valid, x_test = l2_normalize(x_train), l2_normalize(x_valid), l2_normalize(x_test)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    valid_stats = knn_features(x_valid, x_train, y_train, g_train, k_values, args.chunk_size)
    test_stats = knn_features(x_test, x_train, y_train, g_train, k_values, args.chunk_size)
    selected = search_rules(y_valid, valid_stats, args.target_specificity)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "rule" not in selected:
        summary = {
            "status": "NO_GO_FEATURE_GEOMETRY_CLEARANCE",
            "selected": selected,
            "target_specificity": args.target_specificity,
        }
    else:
        rule = selected["rule"]
        valid_clear = apply_rule(valid_stats, rule)
        test_clear = apply_rule(test_stats, rule)
        valid_metrics = metrics(y_valid, valid_clear)
        test_metrics = metrics(y_test, test_clear)
        pred_path = out / "predictions.csv"
        if pred_path.exists():
            pred_path.unlink()
        write_predictions(pred_path, "valid", valid_ids, y_valid, g_valid, valid_clear)
        write_predictions(pred_path, "test", test_ids, y_test, g_test, test_clear)
        go = test_metrics["zero_miss"] and test_metrics["nonreferable_clearance_specificity"] >= args.target_specificity
        summary = {
            "status": "GO_FEATURE_GEOMETRY_90_SPECIFICITY_ZERO_MISS" if go else "NO_GO_FEATURE_GEOMETRY_90_SPECIFICITY_ZERO_MISS",
            "manifest": args.manifest,
            "feature_cache": args.feature_cache,
            "target_specificity": args.target_specificity,
            "l2_normalize": args.l2_normalize,
            "selected_on_validation": selected,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "predictions": str(pred_path),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
