from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import read_yaml_like, write_json


VIEW_SETS = ("macula", "optic_disc", "macula+optic_disc")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def read_splits(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train": set(payload["train_patients"]),
        "calibration": set(payload["calibration_patients"]),
        "validation": set(payload["validation_patients"]),
        "sealed_test": set(payload["sealed_test_patients"]),
    }


def feature_path(cache_dir: Path, image_path: str) -> Path:
    safe = image_path.replace("/", "__").replace(":", "_")
    return cache_dir / f"{safe}.npy"


def image_features(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((image_size, image_size))
        canvas = Image.new("RGB", (image_size, image_size))
        offset = ((image_size - img.width) // 2, (image_size - img.height) // 2)
        canvas.paste(img, offset)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    q10 = np.quantile(flat, 0.10, axis=0)
    q50 = np.quantile(flat, 0.50, axis=0)
    q90 = np.quantile(flat, 0.90, axis=0)
    hist_parts = []
    for c in range(3):
        hist, _ = np.histogram(flat[:, c], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge = np.array(
        [
            float(np.mean(np.abs(gx))),
            float(np.mean(np.abs(gy))),
            float(np.std(gx)),
            float(np.std(gy)),
        ],
        dtype=np.float32,
    )
    yy, xx = np.indices(gray.shape)
    mass = gray + 1e-4
    cx = float((xx * mass).sum() / mass.sum() / image_size)
    cy = float((yy * mass).sum() / mass.sum() / image_size)
    return np.concatenate([means, stds, q10, q50, q90, *hist_parts, edge, np.array([cx, cy])]).astype(np.float32)


def get_feature(cache_dir: Path, root: Path, image_path: str, image_size: int) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = feature_path(cache_dir, image_path)
    if out.exists():
        return np.load(out)
    path = root / image_path
    feat = image_features(path, image_size)
    np.save(out, feat)
    return feat


def group_by_eye(rows: list[dict[str, Any]], cache_dir: Path, root: Path, image_size: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        eye_id = row["eye_id"]
        item = grouped.setdefault(
            eye_id,
            {
                "patient_id": row["patient_id"],
                "eye_id": eye_id,
                "grade": row["grade"],
                "referable_dr": row["referable_dr"],
                "views": {},
            },
        )
        item["views"][row["field"]] = get_feature(cache_dir, root, row["image_path"], image_size)
    return grouped


def assemble_examples(grouped: dict[str, dict[str, Any]], view_set: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    wanted = view_set.split("+")
    ids, xs, ys = [], [], []
    for eye_id, item in sorted(grouped.items()):
        if all(v in item["views"] for v in wanted):
            feats = [item["views"][v] for v in wanted]
            x = np.concatenate([np.mean(feats, axis=0), np.max(feats, axis=0), np.array([len(feats)], dtype=np.float32)])
            ids.append(eye_id)
            xs.append(x)
            ys.append(item["referable_dr"])
    return ids, np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma


def train_logistic(x: np.ndarray, y: np.ndarray, seed: int, epochs: int, lr: float, l2: float) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    b = 0.0
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = neg / max(pos, 1.0)
    weights = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        err = (p - y) * weights
        grad_w = (x.T @ err) / len(y) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict(x: np.ndarray, model: tuple[np.ndarray, float, np.ndarray, np.ndarray]) -> np.ndarray:
    w, b, mu, sigma = model
    return sigmoid(((x - mu) / sigma) @ w + b)


def split_of_patient(patient_id: str, splits: dict[str, set[str]]) -> str:
    for split, patients in splits.items():
        if patient_id in patients:
            return split
    return "unknown"


def patient_folds(patients: list[str], k: int, seed: int) -> list[set[str]]:
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    return [set(shuffled[i::k]) for i in range(k)]


def train_for_view_set(
    grouped: dict[str, dict[str, Any]],
    splits: dict[str, set[str]],
    view_set: str,
    k: int,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
) -> dict[str, tuple[float, str, str]]:
    ids, x, y = assemble_examples(grouped, view_set)
    eye_index = {eye_id: i for i, eye_id in enumerate(ids)}
    non_test_patients = sorted(splits["train"] | splits["calibration"] | splits["validation"])
    folds = patient_folds(non_test_patients, k, seed)
    preds: dict[str, tuple[float, str, str]] = {}

    for fold_id, heldout_patients in enumerate(folds):
        train_ids = [
            eye for eye in ids if grouped[eye]["patient_id"] in set(non_test_patients) - heldout_patients
        ]
        heldout_ids = [eye for eye in ids if grouped[eye]["patient_id"] in heldout_patients]
        if not train_ids or not heldout_ids:
            continue
        train_idx = np.asarray([eye_index[e] for e in train_ids], dtype=int)
        held_idx = np.asarray([eye_index[e] for e in heldout_ids], dtype=int)
        mu, sigma = fit_standardizer(x[train_idx])
        w, b = train_logistic((x[train_idx] - mu) / sigma, y[train_idx], seed + fold_id, epochs, lr, l2)
        model = (w, b, mu, sigma)
        for eye, score in zip(heldout_ids, predict(x[held_idx], model)):
            preds[eye] = (float(score), str(fold_id), "crossfit")

    train_ids = [eye for eye in ids if grouped[eye]["patient_id"] in non_test_patients]
    test_ids = [eye for eye in ids if grouped[eye]["patient_id"] in splits["sealed_test"]]
    if train_ids and test_ids:
        train_idx = np.asarray([eye_index[e] for e in train_ids], dtype=int)
        test_idx = np.asarray([eye_index[e] for e in test_ids], dtype=int)
        mu, sigma = fit_standardizer(x[train_idx])
        w, b = train_logistic((x[train_idx] - mu) / sigma, y[train_idx], seed + 997, epochs, lr, l2)
        model = (w, b, mu, sigma)
        for eye, score in zip(test_ids, predict(x[test_idx], model)):
            preds[eye] = (float(score), "sealed_test_model", "heldout_test")
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cache-dir")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    manifest_path = Path(cfg["dataset"]["manifest_output"])
    split_path = Path(cfg["dataset"]["split_output"])
    output = Path(args.output or cfg["evaluation"]["prediction_input"])
    cache_dir = Path(args.cache_dir or "results/deepdrid_feature_cache")
    seed = int(cfg["splits"]["seed"])

    rows = read_manifest(manifest_path)
    splits = read_splits(split_path)
    grouped = group_by_eye(rows, cache_dir, Path("."), args.image_size)

    predictions_by_view = {
        view_set: train_for_view_set(grouped, splits, view_set, args.folds, seed, args.epochs, args.lr, args.l2)
        for view_set in VIEW_SETS
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "eye_id",
        "split",
        "view_set",
        "grade",
        "referable_dr",
        "referable_score",
        "fold_id",
        "prediction_scope",
        "model_family",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eye_id, item in sorted(grouped.items()):
            for view_set in VIEW_SETS:
                pred = predictions_by_view[view_set].get(eye_id)
                if pred is None:
                    continue
                score, fold_id, scope = pred
                writer.writerow(
                    {
                        "patient_id": item["patient_id"],
                        "eye_id": eye_id,
                        "split": split_of_patient(item["patient_id"], splits),
                        "view_set": view_set,
                        "grade": item["grade"],
                        "referable_dr": item["referable_dr"],
                        "referable_score": f"{score:.8f}",
                        "fold_id": fold_id,
                        "prediction_scope": scope,
                        "model_family": "dependency_light_image_feature_logistic",
                    }
                )

    write_json(
        output.with_suffix(".summary.json"),
        {
            "status": "PREDICTIONS_WRITTEN",
            "output": str(output),
            "model_family": "dependency_light_image_feature_logistic",
            "note": "This is an executable baseline for pipeline readiness. It is not the final pretrained encoder intended for flagship performance claims.",
            "view_sets": VIEW_SETS,
            "n_eyes": len(grouped),
            "folds": args.folds,
            "image_size": args.image_size,
        },
    )


if __name__ == "__main__":
    main()
