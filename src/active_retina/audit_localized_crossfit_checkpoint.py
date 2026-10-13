"""Compare paired and per-view scores for one DRTiD development checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--localization-csv", required=True)
    parser.add_argument("--kaggle-checkpoint", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    localization = Path(args.localization_csv)
    kaggle_checkpoint = Path(args.kaggle_checkpoint)
    model_checkpoint = Path(args.model_checkpoint)
    reference_predictions = Path(args.reference_predictions)
    payload = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    localization_hash = hashlib.sha256(localization.read_bytes()).hexdigest()
    if payload.get("localization_sha256") != localization_hash:
        raise ValueError("Localization hash differs from the one recorded in the checkpoint")
    pairs = read_pairs(manifest, Path.cwd())
    validation_patients = {str(patient) for patient in payload["validation_patient_ids"]}
    valid_pairs = [pair for pair in pairs if str(pair["patient_id"]) in validation_patients]
    observed_patients = {str(pair["patient_id"]) for pair in valid_pairs}
    if observed_patients != validation_patients:
        raise ValueError("Current manifest does not reproduce the checkpoint's frozen validation patients")
    with reference_predictions.open("r", encoding="utf-8", newline="") as stream:
        reference = {row["eye_id"]: (int(row["grade"]), int(row["referable_dr"]))
                     for row in csv.DictReader(stream)}
    current = {pair["eye_id"]: (int(pair["grade"]), int(pair["referable_dr"]))
               for pair in valid_pairs}
    if current != reference:
        raise ValueError("Frozen validation eye IDs/labels differ from the saved training predictions")
    if len(valid_pairs) != len(reference):
        raise ValueError("Duplicate or missing frozen validation eyes in the current manifest")
    dataset = LocalizedPairDataset(valid_pairs, read_optic_disc_centers(localization), args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = LocalizedCrossFieldModel(args.hidden, args.layers).to(args.device)
    checkpoint_info = model.load_kaggle_state(kaggle_checkpoint)
    if checkpoint_info["checkpoint_sha256"] != payload.get("checkpoint_sha256"):
        raise ValueError("Kaggle checkpoint hash differs from the one recorded in the trained checkpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    n = len(valid_pairs)
    labels = np.empty(n, dtype=np.int64)
    grades = np.empty(n, dtype=np.int64)
    scores = {name: np.empty(n, dtype=np.float64) for name in (
        "paired", "macula", "optic_disc", "mean_view", "max_view", "min_view"
    )}
    with torch.inference_mode():
        for images, centers, grade, label, indices in loader:
            logits, view_logits, _residual = model(
                images.to(args.device, non_blocking=True),
                centers.to(args.device, non_blocking=True),
            )
            idx = indices.numpy()
            labels[idx] = label.numpy()
            grades[idx] = grade.numpy()
            view_scores = torch.stack(
                [referable_score(view_logits[:, i]) for i in range(len(FIELDS))], dim=1
            ).float().cpu().numpy()
            paired_scores = referable_score(logits).float().cpu().numpy()
            scores["paired"][idx] = paired_scores
            scores["macula"][idx] = view_scores[:, 0]
            scores["optic_disc"][idx] = view_scores[:, 1]
            scores["mean_view"][idx] = view_scores.mean(axis=1)
            scores["max_view"][idx] = view_scores.max(axis=1)
            scores["min_view"][idx] = view_scores.min(axis=1)

    summary = {
        "status": "EXPLORATORY_SAME_HOLDOUT_SCORE_DIAGNOSTIC",
        "boundary": "All score variants are compared on the repeatedly inspected patient-disjoint internal DRTiD training-partition holdout. The best variant is not an independent or confirmatory result.",
        "n": n,
        "manifest_hash_matches_training": payload.get("manifest_sha256") == manifest_hash,
        "validation_eye_labels_match_training_predictions": True,
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "models": {name: safe_metrics(labels, value) for name, value in scores.items()},
        "epoch": payload.get("epoch"),
        "manifest_sha256": manifest_hash,
        "localization_sha256": localization_hash,
        "kaggle_checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "validation_score_variants.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eye_id", "grade", "referable_dr", *scores])
        for i, pair in enumerate(valid_pairs):
            writer.writerow([pair["eye_id"], int(grades[i]), int(labels[i]),
                             *(float(scores[name][i]) for name in scores)])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
