"""Train a locked ActiveRetina-RC software artifact.

This produces a deterministic software artifact for clinical-style workflow
testing. It is trained on the available DeepDRiD feature cache and includes
binary referable-DR and five-class grading heads for first-view and paired-view
inputs. Regulatory validation is out of scope; software completeness is the aim.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACT_VERSION = "active-retina-rc-clinical-software-2026-10-03"
MANIFEST = Path("results/deepdrid_manifest.csv")
FEATURES = Path("results/deepdrid_resnet50_features.npz")
OUTPUT = Path("models/active_retina_rc_clinical_artifact.joblib")
METADATA = Path("models/active_retina_rc_clinical_artifact.metadata.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def group_by_eye(rows: list[dict[str, Any]], features: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["image_id"] not in features:
            continue
        item = grouped.setdefault(
            row["eye_id"],
            {
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "grade": row["grade"],
                "referable_dr": row["referable_dr"],
                "views": {},
            },
        )
        item["views"][row["field"]] = features[row["image_id"]]
    return [x for x in grouped.values() if "macula" in x["views"] and "optic_disc" in x["views"]]


def pooled(feats: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.mean(feats, axis=0), np.max(feats, axis=0), np.asarray([len(feats)], dtype=np.float32)]).astype(np.float32)


def matrix(items: list[dict[str, Any]], views: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, y_ref, y_grade = [], [], []
    for item in items:
        xs.append(pooled([item["views"][v] for v in views]))
        y_ref.append(item["referable_dr"])
        y_grade.append(item["grade"])
    return np.vstack(xs).astype(np.float32), np.asarray(y_ref, dtype=int), np.asarray(y_grade, dtype=int)


def fit_binary(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=20261003),
    ).fit(x, y)


def fit_grade(x: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=20261004),
    ).fit(x, y)


def main() -> None:
    features = {k: v for k, v in np.load(FEATURES, allow_pickle=False).items()}
    rows = read_manifest(MANIFEST)
    items = group_by_eye(rows, features)
    x_first, y_ref, y_grade = matrix(items, ("macula",))
    x_pair, _, _ = matrix(items, ("macula", "optic_disc"))
    artifact = {
        "version": ARTIFACT_VERSION,
        "encoder": "resnet50.a1_in1k",
        "view_order": {"first": "macula", "second": "optic_disc"},
        "policy": {
            "stop_low": 0.05,
            "stop_high": 0.85,
            "second_low": 0.15,
            "second_high": 0.50,
            "default": "manual_review_on_uncertainty",
        },
        "models": {
            "first_binary": fit_binary(x_first, y_ref),
            "paired_binary": fit_binary(x_pair, y_ref),
            "first_grade": fit_grade(x_first, y_grade),
            "paired_grade": fit_grade(x_pair, y_grade),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT)
    metadata = {
        "version": ARTIFACT_VERSION,
        "status": "SOFTWARE_ARTIFACT_READY_FOR_CLINICAL_WORKFLOW_TESTING_NOT_RISK_CERTIFIED",
        "training_eyes": len(items),
        "manifest_sha256": sha256_file(MANIFEST),
        "feature_cache_sha256": sha256_file(FEATURES),
        "artifact": str(OUTPUT),
        "policy": artifact["policy"],
        "known_limits": [
            "DeepDRiD suffix mapping remains inferred, not official.",
            "Formal 5% missed-DR risk certification is not achieved.",
            "Artifact supports clinical workflow testing, not autonomous diagnosis.",
        ],
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
