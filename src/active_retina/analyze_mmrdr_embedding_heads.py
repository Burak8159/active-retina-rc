"""Compare non-neural decision heads on cached MMRDR development embeddings only."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        splits = data["splits"].astype(str)
        if np.any((splits != "train") & (splits != "valid")):
            raise ValueError(f"Refusing non-development feature cache: {path}")
        if data["model_name"].item() == "" or int(data["image_size"].item()) <= 0:
            raise ValueError(f"Missing feature provenance: {path}")
        return (
            data["features"].astype(np.float32),
            data["labels"].astype(np.int64),
            splits,
            data["image_ids"].astype(str).tolist(),
            data["source_datasets"].astype(str).tolist()
            if "source_datasets" in data.files
            else ["unknown"] * len(splits),
        )


def zero_miss_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    positives = score[y == 1]
    negatives = score[y == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("Validation data must contain both classes")
    threshold = float(positives.min())
    tn = int(np.sum(negatives < threshold))
    fp = int(len(negatives) - tn)
    tp = int(np.sum(positives >= threshold))
    fn = int(len(positives) - tp)
    return {
        "threshold_zero_miss": threshold,
        "n": int(len(y)),
        "positive": int(len(positives)),
        "negative": int(len(negatives)),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "specificity_zero_miss": tn / len(negatives),
        "sensitivity_zero_miss": tp / len(positives),
        "refer_rate": (tp + fp) / len(y),
        "validation_auroc": float(roc_auc_score(y, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-names", nargs="+", help="Optional subset of named heads to refit/export")
    parser.add_argument("--train-sources", nargs="+", help="Optional source_dataset allowlist for training")
    parser.add_argument("--valid-sources", nargs="+", help="Optional source_dataset allowlist for validation")
    args = parser.parse_args()

    caches = [load_cache(path) for path in args.features]
    reference_ids = caches[0][3]
    reference_y, reference_split = caches[0][1:3]
    reference_sources = caches[0][4]
    for path, (_, y, split, ids, sources) in zip(args.features[1:], caches[1:]):
        if (
            ids != reference_ids
            or not np.array_equal(y, reference_y)
            or not np.array_equal(split, reference_split)
            or sources != reference_sources
        ):
            raise ValueError(f"Feature cache identity/order/split mismatch: {path}")

    x_parts = [cache[0] for cache in caches]
    x = np.concatenate(x_parts, axis=1) if len(x_parts) > 1 else x_parts[0]
    train = reference_split == "train"
    valid = reference_split == "valid"
    if args.train_sources:
        train &= np.isin(reference_sources, args.train_sources)
    if args.valid_sources:
        valid &= np.isin(reference_sources, args.valid_sources)
    if not train.any() or not valid.any():
        raise ValueError("Source filters produced an empty train or validation partition")
    y_train, y_valid = reference_y[train], reference_y[valid]
    x_train, x_valid = x[train], x[valid]

    models = []
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        models.append((f"logreg_scaled_c{c:g}", make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, solver="lbfgs"))))
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        models.append((f"linear_svc_scaled_c{c:g}", make_pipeline(StandardScaler(), LinearSVC(C=c, dual="auto", max_iter=5000, random_state=20261013))))
    for c in (0.1, 1.0, 10.0):
        models.append((f"rbf_svc_l2_c{c:g}", make_pipeline(Normalizer(), SVC(C=c, kernel="rbf", probability=False, cache_size=1500))))
    for c in (0.1, 1.0, 10.0):
        models.append((f"rbf_svc_scaled_c{c:g}", make_pipeline(StandardScaler(), PCA(n_components=min(128, x_train.shape[1], len(x_train) - 1), random_state=20261013), SVC(C=c, kernel="rbf", probability=False, cache_size=1500))))
    models.extend([
        ("extra_trees", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, class_weight=None, random_state=20261013)),
        ("random_forest", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, class_weight=None, random_state=20261013)),
        ("hist_gradient_boost", HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.08, random_state=20261013)),
    ])
    if args.model_names:
        selected = dict(models)
        unknown = sorted(set(args.model_names) - set(selected))
        if unknown:
            parser.error(f"unknown model name(s): {', '.join(unknown)}")
        models = [(name, selected[name]) for name in args.model_names]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    score_rows = []
    for name, model in models:
        model.fit(x_train, y_train)
        score = model.decision_function(x_valid) if hasattr(model, "decision_function") else model.predict_proba(x_valid)[:, 1]
        score = np.asarray(score, dtype=float)
        results.append({"model": name, **zero_miss_metrics(y_valid, score)})
        score_rows.extend(
            {"image_id": reference_ids[index], "referable_dr": int(y_valid[position]), "model": name, "score": float(value)}
            for position, (index, value) in enumerate(zip(np.flatnonzero(valid), score))
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    results.sort(key=lambda row: (float(row["specificity_zero_miss"]), float(row["validation_auroc"])), reverse=True)
    payload = {
        "status": "DEVELOPMENT_VALIDATION_ONLY_OFFICIAL_TEST_NOT_READ",
        "dataset": "MMRDR mixed-source development transfer" if len(set(reference_sources)) > 1 else "MMRDR",
        "binary_rule": "grade >= 2 is referable; grade 0-1 is nonreferable",
        "split_caveat": "CFP and UWF are independent cohorts, not filename-paired views; CFP official test and UWF official test were excluded; UWF internal train/validation is image-level because patient IDs are unavailable in the exported manifest",
        "source_counts": dict(Counter(reference_sources)),
        "train_source_counts": dict(Counter(source for source, keep in zip(reference_sources, train) if keep)),
        "validation_source_counts": dict(Counter(source for source, keep in zip(reference_sources, valid) if keep)),
        "feature_sources": [str(path) for path in args.features],
        "n_train": int(train.sum()),
        "n_valid": int(valid.sum()),
        "results_ranked_by_zero_miss_specificity": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output_dir / "validation_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "referable_dr", "model", "score"])
        writer.writeheader()
        writer.writerows(score_rows)


if __name__ == "__main__":
    main()
