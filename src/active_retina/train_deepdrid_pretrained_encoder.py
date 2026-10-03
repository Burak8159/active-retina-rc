from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .common import read_yaml_like, write_json


VIEW_SETS = ("macula", "optic_disc", "macula+optic_disc")


class ImagePathDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], transform: Any) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_id, path = self.items[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return image_id, self.transform(image)


def build_encoder(model_name: str, pretrained: bool):
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    model.eval()
    return model, transform


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


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


def extract_features(
    rows: list[dict[str, Any]],
    cache_path: Path,
    model_name: str,
    pretrained: bool,
    batch_size: int,
    device_name: str,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return {k: data[k] for k in data.files}

    seen = {}
    for row in rows:
        seen[row["image_id"]] = row["image_path"]
    items = sorted(seen.items())

    device = device_for(device_name)
    model, transform = build_encoder(model_name, pretrained)
    model.to(device)
    dataset = ImagePathDataset(items, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for image_ids, batch in loader:
            batch = batch.to(device)
            out = model(batch)
            out = out.detach().cpu().float().numpy()
            for image_id, feat in zip(image_ids, out):
                feats[str(image_id)] = feat.astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **feats)
    return feats


def group_by_eye(rows: list[dict[str, Any]], features: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["image_id"] not in features:
            continue
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
        item["views"][row["field"]] = features[row["image_id"]]
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
    return ids, np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=int)


def split_of_patient(patient_id: str, splits: dict[str, set[str]]) -> str:
    for split, patients in splits.items():
        if patient_id in patients:
            return split
    return "unknown"


def patient_folds(patients: list[str], k: int, seed: int) -> list[set[str]]:
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    return [set(shuffled[i::k]) for i in range(k)]


def fit_model(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    model = make_pipeline(StandardScaler(), clf)
    model.fit(x, y)
    return model


def train_for_view_set(
    grouped: dict[str, dict[str, Any]],
    splits: dict[str, set[str]],
    view_set: str,
    k: int,
    seed: int,
    include_test: bool = False,
) -> dict[str, tuple[float, str, str]]:
    ids, x, y = assemble_examples(grouped, view_set)
    eye_index = {eye_id: i for i, eye_id in enumerate(ids)}
    non_test_patients = sorted(splits["train"])
    folds = patient_folds(non_test_patients, k, seed)
    preds: dict[str, tuple[float, str, str]] = {}

    for fold_id, heldout_patients in enumerate(folds):
        train_ids = [eye for eye in ids if grouped[eye]["patient_id"] in set(non_test_patients) - heldout_patients]
        heldout_ids = [eye for eye in ids if grouped[eye]["patient_id"] in heldout_patients]
        if not train_ids or not heldout_ids:
            continue
        train_idx = np.asarray([eye_index[e] for e in train_ids], dtype=int)
        held_idx = np.asarray([eye_index[e] for e in heldout_ids], dtype=int)
        model = fit_model(x[train_idx], y[train_idx], seed + fold_id)
        scores = model.predict_proba(x[held_idx])[:, 1]
        for eye, score in zip(heldout_ids, scores):
            preds[eye] = (float(score), str(fold_id), "crossfit")

    train_ids = [eye for eye in ids if grouped[eye]["patient_id"] in non_test_patients]
    heldout_patients = splits["calibration"] | splits["validation"]
    if include_test:
        heldout_patients = heldout_patients | splits["sealed_test"]
    test_ids = [eye for eye in ids if grouped[eye]["patient_id"] in heldout_patients]
    if train_ids and test_ids:
        train_idx = np.asarray([eye_index[e] for e in train_ids], dtype=int)
        test_idx = np.asarray([eye_index[e] for e in test_ids], dtype=int)
        model = fit_model(x[train_idx], y[train_idx], seed + 997)
        scores = model.predict_proba(x[test_idx])[:, 1]
        for eye, score in zip(test_ids, scores):
            split = split_of_patient(grouped[eye]["patient_id"], splits)
            preds[eye] = (float(score), "train_only_model", "heldout_test" if split == "sealed_test" else "heldout_development")
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-test", action="store_true", help="Explicit final evaluation only; omitted during development.")
    args = parser.parse_args()

    cfg = read_yaml_like(Path(args.config))
    manifest_path = Path(cfg["dataset"]["manifest_output"])
    split_path = Path(cfg["dataset"]["split_output"])
    output = Path(args.output or cfg["evaluation"]["prediction_input"].replace(".csv", "_pretrained.csv"))
    seed = int(cfg["splits"]["seed"])

    rows = read_manifest(manifest_path)
    splits = read_splits(split_path)
    memberships = [patients for patients in splits.values()]
    if any(a & b for i, a in enumerate(memberships) for b in memberships[i + 1:]):
        raise ValueError("Patient splits overlap")
    if not args.include_test:
        rows = [row for row in rows if row["patient_id"] not in splits["sealed_test"]]
    features = extract_features(
        rows,
        Path(args.feature_cache),
        args.model_name,
        not args.no_pretrained,
        args.batch_size,
        args.device,
    )
    grouped = group_by_eye(rows, features)
    predictions_by_view = {
        view_set: train_for_view_set(grouped, splits, view_set, args.folds, seed, args.include_test)
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
        "encoder",
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
                        "model_family": "frozen_pretrained_encoder_logistic",
                        "encoder": args.model_name,
                    }
                )

    write_json(
        output.with_suffix(".summary.json"),
        {
            "status": "PRETRAINED_ENCODER_PREDICTIONS_WRITTEN",
            "output": str(output),
            "model_family": "frozen_pretrained_encoder_logistic",
            "encoder": args.model_name,
            "pretrained": not args.no_pretrained,
            "feature_cache": args.feature_cache,
            "view_sets": VIEW_SETS,
            "n_eyes": len(grouped),
            "folds": args.folds,
            "training_patients": "train_only",
            "test_included": args.include_test,
        },
    )


if __name__ == "__main__":
    main()
