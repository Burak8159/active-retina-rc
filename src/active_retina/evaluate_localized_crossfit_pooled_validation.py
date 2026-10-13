"""Exploratory evaluation of one frozen DRTiD model on pooled validation rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_drtid_localized_crossfit import (
    LocalizedCrossFieldModel,
    LocalizedPairDataset,
    read_optic_disc_centers,
)
from .train_drtid_paired_fusion import FIELDS, read_pairs, referable_score, safe_metrics


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_split_pairs(manifest: Path, root: Path, split_name: str):
    grouped = defaultdict(dict)
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("source_dataset") != "drtid_paired" or row.get("source_split") != split_name:
                continue
            eye = grouped[row["eye_id"]]
            if eye and (
                eye["patient_id"] != row["patient_id"]
                or eye["grade"] != int(row["grade"])
                or eye["referable_dr"] != int(row["referable_dr"])
            ):
                raise ValueError(f"Conflicting rows for eye {row['eye_id']}")
            eye.update({
                "patient_id": row["patient_id"],
                "eye_id": row["eye_id"],
                "grade": int(row["grade"]),
                "referable_dr": int(row["referable_dr"]),
                "views": {**eye.get("views", {}), row["field"]: row["image_path"]},
            })
    pairs = [item for item in grouped.values() if all(field in item["views"] for field in FIELDS)]
    if not pairs:
        raise ValueError(f"No complete DRTiD pairs in source split {split_name!r}")
    for pair in pairs:
        for field in FIELDS:
            path = Path(pair["views"][field])
            pair["views"][field] = str(path if path.is_file() else root / path)
            if not Path(pair["views"][field]).is_file():
                raise FileNotFoundError(pair["views"][field])
    return pairs


def fixed_threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float):
    positive, negative = labels == 1, labels == 0
    action_clear = scores < threshold
    return {
        "threshold_from_internal_calibration": float(threshold),
        "n": int(len(labels)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "misses": int(np.sum(positive & action_clear)),
        "true_negatives": int(np.sum(negative & action_clear)),
        "specificity": float(np.mean(action_clear[negative])) if negative.any() else None,
        "coverage": 1.0,
    }


def infer_scores(model, pairs, centers, args):
    dataset = LocalizedPairDataset(pairs, centers, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    grades, labels = np.empty(len(pairs), dtype=np.int64), np.empty(len(pairs), dtype=np.int64)
    score_names = ("paired", "macula", "optic_disc", "mean_view", "max_view", "min_view")
    scores = {key: np.empty(len(pairs), dtype=np.float64) for key in score_names}
    feature_names = tuple(
        [f"paired_grade_{grade}" for grade in range(5)]
        + [f"view{view}_grade_{grade}" for view in range(2) for grade in range(5)]
    )
    features = {key: np.empty(len(pairs), dtype=np.float64) for key in feature_names}
    with torch.inference_mode():
        for images, pair_centers, grade, label, indices in loader:
            images = images.to(args.device, non_blocking=True)
            pair_centers = pair_centers.to(args.device, non_blocking=True)
            if args.tta != "none":
                fused_logits, per_view_logits = [], []
                transforms = {
                    "hflip": ((False, False, 0), (True, False, 0)),
                    "vflip": ((False, False, 0), (False, True, 0)),
                    "flips": ((False, False, 0), (True, False, 0),
                              (False, True, 0), (True, True, 0)),
                    "d4": tuple((flip_x, False, rotation)
                                for flip_x in (False, True) for rotation in range(4)),
                }[args.tta]
                for flip_x, flip_y, rotation in transforms:
                    transformed_images = images
                    transformed_centers = pair_centers.clone()
                    if flip_x:
                        transformed_images = torch.flip(transformed_images, dims=(-1,))
                        transformed_centers[..., 0] = 1.0 - transformed_centers[..., 0]
                    if flip_y:
                        transformed_images = torch.flip(transformed_images, dims=(-2,))
                        transformed_centers[..., 1] = 1.0 - transformed_centers[..., 1]
                    if rotation:
                        transformed_images = torch.rot90(
                            transformed_images, k=rotation, dims=(-2, -1)
                        )
                        x, y = transformed_centers[..., 0].clone(), transformed_centers[..., 1].clone()
                        if rotation == 1:
                            transformed_centers[..., 0], transformed_centers[..., 1] = y, 1.0 - x
                        elif rotation == 2:
                            transformed_centers[..., 0], transformed_centers[..., 1] = 1.0 - x, 1.0 - y
                        else:
                            transformed_centers[..., 0], transformed_centers[..., 1] = 1.0 - y, x
                    fused, views, _residual = model(transformed_images, transformed_centers)
                    fused_logits.append(fused)
                    per_view_logits.append(views)
                logits = torch.stack(fused_logits).mean(dim=0)
                view_logits = torch.stack(per_view_logits).mean(dim=0)
            else:
                logits, view_logits, _residual = model(images, pair_centers)
            idx = indices.numpy()
            grades[idx], labels[idx] = grade.numpy(), label.numpy()
            per_view = torch.stack(
                [referable_score(view_logits[:, i]) for i in range(2)], dim=1
            ).float().cpu().numpy()
            score_values = {
                "paired": referable_score(logits).float().cpu().numpy(),
                "macula": per_view[:, 0],
                "optic_disc": per_view[:, 1],
                "mean_view": per_view.mean(axis=1),
                "max_view": per_view.max(axis=1),
                "min_view": per_view.min(axis=1),
            }
            for name, value in score_values.items():
                scores[name][idx] = value
            feature_values = {
                **{f"paired_grade_{grade}": torch.softmax(logits.float(), dim=1)[:, grade].cpu().numpy()
                   for grade in range(5)},
                **{f"view{view}_grade_{grade}": torch.softmax(view_logits[:, view].float(), dim=1)[:, grade].cpu().numpy()
                   for view in range(2) for grade in range(5)},
            }
            for name, value in feature_values.items():
                features[name][idx] = value
    return grades, labels, scores, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--kaggle-checkpoint", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--calibration-predictions")
    parser.add_argument("--ceiling-json", required=True)
    parser.add_argument("--source-split", default="drtid_paired_valid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", choices=("none", "hflip", "vflip", "flips", "d4"), default="none")
    args = parser.parse_args()

    manifest, localization = Path(args.manifest), Path(args.localization_csv)
    model_checkpoint, kaggle_checkpoint = Path(args.model_checkpoint), Path(args.kaggle_checkpoint)
    payload = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    model = LocalizedCrossFieldModel(args.hidden, args.layers).to(args.device)
    checkpoint_info = model.load_kaggle_state(kaggle_checkpoint)
    if checkpoint_info["checkpoint_sha256"] != payload.get("checkpoint_sha256"):
        raise ValueError("Kaggle checkpoint hash differs from the trained artifact")
    if hashlib.sha256(localization.read_bytes()).hexdigest() != payload.get("localization_sha256"):
        raise ValueError("Localization file hash differs from the trained artifact")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    pairs = read_split_pairs(manifest, Path.cwd(), args.source_split)
    canonical_patient = lambda value: str(value).rsplit(":", 1)[-1]
    training_patients = {canonical_patient(value) for value in payload.get("train_patient_ids", [])}
    evaluation_patients = {canonical_patient(pair["patient_id"]) for pair in pairs}
    patient_overlap = sorted(training_patients & evaluation_patients)
    if patient_overlap:
        raise ValueError(f"Training/evaluation patient overlap detected ({len(patient_overlap)} IDs)")
    all_train_pairs = read_pairs(manifest, Path.cwd())
    calibration_patient_ids = set(payload.get("validation_patient_ids", []))
    calibration_pairs = [pair for pair in all_train_pairs if pair["patient_id"] in calibration_patient_ids]
    found_calibration_ids = {pair["patient_id"] for pair in calibration_pairs}
    if not calibration_patient_ids or found_calibration_ids != calibration_patient_ids:
        raise ValueError("Checkpoint calibration patients do not match the current manifest training pool")
    calibration_patients = {canonical_patient(pair["patient_id"]) for pair in calibration_pairs}
    if calibration_patients & evaluation_patients:
        raise ValueError("Calibration/evaluation patient overlap detected")
    centers = read_optic_disc_centers(localization)
    calibration_grades, calibration_labels, calibration_scores, calibration_features = infer_scores(
        model, calibration_pairs, centers, args
    )
    if args.calibration_predictions:
        recorded_calibration = read_csv(Path(args.calibration_predictions))
        recorded_by_eye = {row["eye_id"]: row for row in recorded_calibration}
        calibration_by_eye = {pair["eye_id"]: i for i, pair in enumerate(calibration_pairs)}
        if set(recorded_by_eye) != set(calibration_by_eye):
            raise ValueError("Recorded calibration predictions do not match checkpoint calibration eyes")
        for eye_id, index in calibration_by_eye.items():
            row = recorded_by_eye[eye_id]
            if int(row["referable_dr"]) != int(calibration_labels[index]):
                raise ValueError(f"Calibration label mismatch for {eye_id}")
            if args.tta == "none" and not np.isclose(
                float(row["score"]), calibration_scores["paired"][index], rtol=1e-5, atol=1e-5
            ):
                raise ValueError(f"Calibration score mismatch for {eye_id}")
    grades, labels, scores, features = infer_scores(model, pairs, centers, args)
    score_names = tuple(scores)
    n = len(pairs)

    ceiling = json.loads(Path(args.ceiling_json).read_text(encoding="utf-8"))
    baseline = ceiling["best_by_source"]
    expected = baseline["drtid_paired"]["metrics"]
    negative_count = int(np.sum(labels == 0))
    positive_count = int(np.sum(labels == 1))
    other_true_negatives = sum(
        int(item["metrics"]["gradable_nonreferable"]
            - item["metrics"]["false_referrals_on_gradable_nonreferable"])
        for source, item in baseline.items() if source != "drtid_paired"
    )
    aggregate_negatives = int(ceiling["aggregate"]["gradable_nonreferable"])
    output_metrics = {name: safe_metrics(labels, values) for name, values in scores.items()}
    calibration_selection = {}
    frozen_threshold_metrics = {}
    for name in score_names:
        metrics = safe_metrics(calibration_labels, calibration_scores[name])
        threshold = float(metrics["threshold_zero_miss"])
        calibration_selection[name] = metrics
        frozen_threshold_metrics[name] = fixed_threshold_metrics(labels, scores[name], threshold)
    selected_score = max(
        score_names,
        key=lambda name: (
            float(calibration_selection[name]["specificity_at_zero_miss"]),
            float(calibration_selection[name].get("auroc", 0.0)),
        ),
    )
    selected_threshold = float(calibration_selection[selected_score]["threshold_zero_miss"])
    fixed_threshold = {
        "selected_score_from_internal_calibration": selected_score,
        "selected_threshold_from_internal_calibration": selected_threshold,
        "calibration_metrics": calibration_selection[selected_score],
        "pooled_validation_metrics": frozen_threshold_metrics[selected_score],
    }
    pooled = {}
    for name, metrics in output_metrics.items():
        if (negative_count, positive_count) != (
            int(expected["gradable_nonreferable"]), int(expected["gradable_referable_dr"])
        ):
            raise ValueError("This DRTiD split does not match the pooled ceiling's DRTiD cohort size")
        tn = int(round(float(metrics["specificity_at_zero_miss"]) * negative_count))
        pooled[name] = {
            "drtid_true_negatives": tn,
            "drtid_negatives": negative_count,
            "drtid_misses": int(metrics["misses_at_threshold"]),
            "pooled_true_negatives": other_true_negatives + tn,
            "pooled_negatives": aggregate_negatives,
            "pooled_specificity": (other_true_negatives + tn) / aggregate_negatives,
        }
    summary = {
        "status": "EXPOSED_VALIDATION_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "boundary": "The DRTiD validation cohort was already used in prior source-expert selection. Zero-miss thresholds below are set from this same cohort. Results are exploratory and cannot certify risk or generalization.",
        "source_split": args.source_split,
        "n": n,
        "training_patient_count_from_checkpoint": len(training_patients),
        "evaluation_patient_count": len(evaluation_patients),
        "training_evaluation_patient_overlap": len(patient_overlap),
        "calibration_patient_count": len(calibration_patients),
        "calibration_evaluation_patient_overlap": len(calibration_patients & evaluation_patients),
        "positive": positive_count,
        "negative": negative_count,
        "grade_counts": {str(grade): int((grades == grade).sum()) for grade in sorted(set(grades.tolist()))},
        "candidate_epoch": payload.get("epoch"),
        "test_time_augmentation": args.tta,
        "candidate_training_manifest_sha256": payload.get("manifest_sha256"),
        "evaluation_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "score_metrics": output_metrics,
        "internal_calibration_score_selection": calibration_selection,
        "frozen_calibrated_threshold_metrics_by_score": frozen_threshold_metrics,
        "internal_calibration_threshold_metrics_on_pooled_validation": fixed_threshold,
        "pooled_counterfactual_if_replaced_drtid_expert": pooled,
        "baseline_other_source_true_negatives": other_true_negatives,
        "baseline_drtid": expected,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "patient_id", "grade", "referable_dr", *score_names, *features])
        for i, pair in enumerate(pairs):
            writer.writerow([pair["eye_id"], pair["patient_id"], int(grades[i]), int(labels[i]),
                             *(float(scores[name][i]) for name in score_names),
                             *(float(features[name][i]) for name in features)])
    with (output / "calibration_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "patient_id", "grade", "referable_dr", *score_names,
                         *calibration_features])
        for i, pair in enumerate(calibration_pairs):
            writer.writerow([pair["eye_id"], pair["patient_id"], int(calibration_grades[i]),
                             int(calibration_labels[i]),
                             *(float(calibration_scores[name][i]) for name in score_names),
                             *(float(calibration_features[name][i]) for name in calibration_features)])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
