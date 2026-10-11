"""Train compact risk models on frozen RETFound eye-level embeddings.

Only patient-disjoint train/validation rows are loaded. Validation selects the
zero-miss binary operating point; the development test split is never opened.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "vit_large_patch16_224"
CHECKPOINT_GLOB = "external/model_cache/hub/models--bswift--RETfound_eyepacs_DR/snapshots/*/model.safetensors"


def read_development_rows(manifest: Path) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = str(row["source_split"]).lower().rsplit("_", 1)[-1]
            if split not in splits:
                continue
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            splits[split].append(row)
    if not splits["train"] or not splits["valid"]:
        raise ValueError("Manifest must include nonempty patient-disjoint train and valid rows")
    return splits


def resolve_image(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    normalized = raw_path.replace("\\", "/")
    if "/external/" in normalized:
        path = root / "external" / normalized.split("/external/", 1)[1]
    else:
        path = root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def grouped_eyes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        eye = grouped.setdefault(str(row["eye_id"]), {
            "eye_id": str(row["eye_id"]),
            "patient_id": str(row["patient_id"]),
            "source_dataset": str(row["source_dataset"]),
            "grade": int(row["grade"]),
            "referable": int(row["referable_dr"]),
            "views": [],
        })
        for key, value in (("patient_id", str(row["patient_id"])), ("source_dataset", str(row["source_dataset"])),
                           ("grade", int(row["grade"])), ("referable", int(row["referable_dr"]))):
            if eye[key] != value:
                raise ValueError(f"Conflicting {key} labels within eye {eye['eye_id']}")
        eye["views"].append(row)
    return list(grouped.values())


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, root: Path) -> None:
        self.rows, self.transform, self.root = rows, transform, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(resolve_image(self.root, row["image_path"])) as image:
            image = image.convert("L").convert("RGB")
            return self.transform(image), str(row["eye_id"])


def load_model(checkpoint: Path, device: torch.device):
    import timm
    from safetensors.torch import load_file

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5, img_size=256, global_pool="avg")
    incompat = model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"RETFound checkpoint mismatch: {incompat}")
    config = timm.data.resolve_model_data_config(model)
    config["input_size"] = (3, 256, 256)
    transform = timm.data.create_transform(**config, is_training=False)
    model.to(device).eval()
    return model, transform


def extract_eye_features(model: torch.nn.Module, transform: Any, eyes: list[dict[str, Any]], root: Path,
                         device: torch.device, batch_size: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for eye in eyes for row in eye["views"]]
    loader = DataLoader(ImageRows(rows, transform, root), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    per_eye: dict[str, list[np.ndarray]] = defaultdict(list)
    per_eye_scores: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for images, eye_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                encoded = model.forward_features(images)
                features = model.forward_head(encoded, pre_logits=True)
                logits = model.forward_head(encoded)
                per_view_score = torch.softmax(logits.float(), dim=1)[:, 2:5].sum(dim=1)
            for eye_id, feature, score in zip(eye_ids, features.float().cpu().numpy(), per_view_score.cpu().numpy()):
                per_eye[str(eye_id)].append(feature.astype(np.float32, copy=False))
                per_eye_scores[str(eye_id)].append(float(score))
    output: list[np.ndarray] = []
    pooled_scores: list[float] = []
    for eye in eyes:
        views = np.stack(per_eye[eye["eye_id"]])
        mean = views.mean(axis=0)
        maximum = views.max(axis=0)
        difference = np.abs(views[0] - views[1]) if len(views) > 1 else np.zeros_like(mean)
        output.append(np.concatenate((mean, maximum, difference)).astype(np.float32))
        pooled_scores.append(max(per_eye_scores[eye["eye_id"]]))
    return np.stack(output), np.asarray(pooled_scores, dtype=float)


def zero_miss_metrics(scores: np.ndarray, eyes: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    grades = np.asarray([eye["grade"] for eye in eyes])
    labels = np.asarray([eye["referable"] for eye in eyes])
    sources = np.asarray([eye["source_dataset"] for eye in eyes])
    eligible = grades < 5
    negative = eligible & (labels == 0)
    positive = eligible & (labels == 1)
    # Every action is based on the model score. Ground-truth grade must never
    # decide an output, including whether a known ungradable case is referred.
    referred = scores >= threshold
    missed = positive & ~referred
    result: dict[str, Any] = {
        "n_eyes": len(eyes),
        "n_gradable": int(eligible.sum()),
        "n_gradable_nonreferable": int(negative.sum()),
        "n_referable": int(positive.sum()),
        "specificity": float((~referred[negative]).mean()) if negative.any() else None,
        "true_negative_clearances": int((~referred[negative]).sum()),
        "false_negative_clearances": int(missed.sum()),
        "binary_action_coverage_including_ungradable_referral": 1.0,
        "ungradable_referral_rate": float(referred[grades == 5].mean()) if np.any(grades == 5) else None,
        "referral_rate": float(referred.mean()),
        "threshold": float(threshold),
        "by_source": {},
        "by_grade": {},
    }
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        neg = mask & negative
        pos = mask & positive
        result["by_source"][source] = {
            "negative_n": int(neg.sum()),
            "specificity": float((~referred[neg]).mean()) if neg.any() else None,
            "referable_n": int(pos.sum()),
            "misses": int(missed[pos].sum()),
        }
    for grade in sorted(set(grades.tolist())):
        mask = grades == grade
        neg = mask & negative
        pos = mask & positive
        result["by_grade"][str(grade)] = {
            "n": int(mask.sum()),
            "specificity": float((~referred[neg]).mean()) if neg.any() else None,
            "misses": int(missed[pos].sum()),
            "referral_rate": float(referred[mask].mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path.cwd()
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(sorted(glob.glob(str(root / CHECKPOINT_GLOB)))[-1])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, transform = load_model(checkpoint, device)
    splits = read_development_rows(Path(args.manifest))
    train_eyes, valid_eyes = grouped_eyes(splits["train"]), grouped_eyes(splits["valid"])
    x_train, train_pretrained_risk = extract_eye_features(model, transform, train_eyes, root, device, args.batch_size, args.num_workers)
    x_valid, valid_pretrained_risk = extract_eye_features(model, transform, valid_eyes, root, device, args.batch_size, args.num_workers)
    y_train = np.asarray([eye["referable"] for eye in train_eyes])
    train_grades = np.asarray([eye["grade"] for eye in train_eyes])
    fit_mask = train_grades < 5
    y_fit, x_fit = y_train[fit_mask], x_train[fit_mask]
    y_valid = np.asarray([eye["referable"] for eye in valid_eyes])
    valid_grades = np.asarray([eye["grade"] for eye in valid_eyes])
    valid_mask = valid_grades < 5
    candidates = {
        "logistic_c0001": make_pipeline(StandardScaler(), LogisticRegression(C=0.001, class_weight="balanced", max_iter=3000)),
        "logistic_c001": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000)),
        "logistic_c01": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        "linear_svc_c001": make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight="balanced", dual="auto", max_iter=5000)),
        "rbf_svc_c001": make_pipeline(StandardScaler(), SVC(C=0.01, kernel="rbf", class_weight="balanced")),
        "rbf_svc_c01": make_pipeline(StandardScaler(), SVC(C=0.1, kernel="rbf", class_weight="balanced")),
    }
    outcomes: dict[str, Any] = {}
    valid_threshold_scores = valid_pretrained_risk.copy()
    valid_threshold_scores[~valid_mask] = -np.inf
    raw_positive = valid_threshold_scores[y_valid == 1]
    raw_threshold = float(np.min(raw_positive) - np.finfo(np.float64).eps * max(1.0, abs(float(np.min(raw_positive)))))
    outcomes["retfound_pretrained_max_view_referable_probability"] = zero_miss_metrics(
        valid_pretrained_risk, valid_eyes, raw_threshold
    )
    for name, estimator in candidates.items():
        estimator.fit(x_fit, y_fit)
        score = np.asarray(estimator.decision_function(x_valid), dtype=float)
        positive_scores = score[(y_valid == 1) & valid_mask]
        threshold = float(np.min(positive_scores) - np.finfo(np.float64).eps * max(1.0, abs(float(np.min(positive_scores)))))
        outcomes[name] = zero_miss_metrics(score, valid_eyes, threshold)
    train_multiclass = fit_mask
    for name, c_value, balanced in (
        ("ordinal_grade_logistic_c001", 0.01, False),
        ("ordinal_grade_logistic_c001_balanced", 0.01, True),
        ("ordinal_grade_logistic_c01", 0.1, False),
        ("ordinal_grade_logistic_c01_balanced", 0.1, True),
    ):
        ordinal_model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c_value, class_weight="balanced" if balanced else None, max_iter=3000
        ))
        ordinal_model.fit(x_train[train_multiclass], train_grades[train_multiclass])
        probabilities = ordinal_model.predict_proba(x_valid)
        ordinal_classes = ordinal_model[-1].classes_.astype(int)
        score = probabilities[:, np.isin(ordinal_classes, (2, 3, 4))].sum(axis=1)
        positive_scores = score[(y_valid == 1) & valid_mask]
        threshold = float(np.min(positive_scores) - np.finfo(np.float64).eps * max(1.0, abs(float(np.min(positive_scores)))))
        outcomes[name] = zero_miss_metrics(score, valid_eyes, threshold)
    ranking = sorted(outcomes, key=lambda name: (
        outcomes[name]["specificity"] if outcomes[name]["specificity"] is not None else -1.0,
        -outcomes[name]["referral_rate"],
    ), reverse=True)
    report = {
        "status": "VALIDATION_ONLY_DEVELOPMENT_DIAGNOSTIC",
        "claim_boundary": "All project datasets have been inspected; this is not untouched certification. Only train and validation images were loaded; test split was not opened.",
        "backbone": "RETFound ViT-L/16 (EyePACS-pretrained checkpoint), strict-loaded into timm vit_large_patch16_224 at 256px with average pooling",
        "checkpoint": str(checkpoint),
        "head_training": "train-split binary and 5-grade classifiers on concatenated mean, max, and absolute-difference eye embeddings; grade-5 DDR cases are excluded from fitting, and all outputs including grade-5 predictions are determined only by model scores",
        "n_train_eyes": len(train_eyes),
        "n_validation_eyes": len(valid_eyes),
        "n_fitted_gradable_train_eyes": int(fit_mask.sum()),
        "n_validation_gradable_eyes": int(valid_mask.sum()),
        "selection_rule": "Within each candidate, choose the largest clearance threshold that yields zero missed gradable referable validation eyes; rank candidates by validation specificity then lower referral rate. Ungradable referrals are reported from model output, not forced by labels.",
        "candidate_results": outcomes,
        "ranked_candidates": ranking,
        "best_candidate": ranking[0],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"best_candidate": report["best_candidate"],
                      "best_validation_metrics": outcomes[report["best_candidate"]],
                      "ranked_candidates": ranking}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
