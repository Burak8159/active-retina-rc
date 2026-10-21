"""Score a frozen MMRDR candidate once on the official test-only manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_every_image_autonomy import FundusDataset, GradeReferableModel, metrics, predict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def one_sided_miss_risk_upper(misses: int, positives: int, alpha: float = 0.05) -> float:
    if positives <= 0:
        return 1.0
    if misses == 0:
        return float(1.0 - alpha ** (1.0 / positives))
    from scipy.stats import beta

    return float(beta.ppf(1.0 - alpha, misses + 1, positives - misses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-before-inference",
        action="store_true",
        help="Resume the same locked attempt only if inference never started and no result files exist.",
    )
    args = parser.parse_args()

    with args.test_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("source_split") != "test" for row in rows):
        raise SystemExit("Refusing to evaluate: input must contain official test rows only.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test_evaluated") is not False:
        raise SystemExit("Checkpoint lacks the test-not-yet-evaluated marker; refusing to call this a one-shot test.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "OFFICIAL_TEST_EVALUATION_STARTED.lock"
    input_record = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "test_manifest": str(args.test_manifest.resolve()),
        "test_manifest_sha256": sha256_file(args.test_manifest),
    }
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(json.dumps(input_record, sort_keys=True) + "\n")
    except FileExistsError as exc:
        inference_lock = args.output_dir / "INFERENCE_STARTED.lock"
        result_paths = (args.output_dir / "test_predictions.csv", args.output_dir / "summary.json")
        try:
            prior_attempt = json.loads(lock_path.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, json.JSONDecodeError, IndexError) as read_exc:
            raise SystemExit(f"Existing one-shot lock is unreadable; refusing to resume: {lock_path}") from read_exc
        same_inputs = (
            prior_attempt.get("checkpoint") == input_record["checkpoint"]
            and prior_attempt.get("checkpoint_sha256") == input_record["checkpoint_sha256"]
            and prior_attempt.get("test_manifest") == input_record["test_manifest"]
            and prior_attempt.get("test_manifest_sha256") == input_record["test_manifest_sha256"]
        )
        if (
            not args.resume_before_inference
            or not same_inputs
            or inference_lock.exists()
            or any(path.exists() for path in result_paths)
        ):
            raise SystemExit(f"One-shot test was already started or cannot be safely resumed: {lock_path}") from exc
    device = torch.device(args.device) if args.device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    wrapper = GradeReferableModel(checkpoint["model_name"], checkpoint.get("args", {}).get("image_size", 224), pretrained=False)
    model = wrapper.model
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    dataset = FundusDataset(rows, wrapper.eval_transform, Path.cwd())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    inference_lock = args.output_dir / "INFERENCE_STARTED.lock"
    try:
        with inference_lock.open("x", encoding="utf-8") as marker:
            marker.write("Inference began; this test attempt cannot be resumed.\n")
    except FileExistsError as exc:
        raise SystemExit(f"Inference-start marker already exists; refusing a second pass: {inference_lock}") from exc
    labels, grades, scores, ids = predict(model, loader, device)
    threshold = float(checkpoint["threshold"])
    result = metrics(labels, scores, threshold)
    negatives = result["negative"]
    positives = result["positive"]
    specificity_ci = wilson_interval(result["true_negative"], negatives)
    missed_risk_ucb = one_sided_miss_risk_upper(result["false_negative_clearance"], positives)
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "grade", "referable_dr", "score", "threshold", "action"])
        writer.writeheader()
        for image_id, grade, label, score in zip(ids, grades, labels, scores):
            writer.writerow({"image_id": image_id, "grade": int(grade), "referable_dr": int(label), "score": f"{float(score):.8f}", "threshold": f"{threshold:.8f}", "action": "REFER" if score >= threshold else "CLEAR"})
    summary: dict[str, Any] = {
        "status": "MMRDR_OFFICIAL_TEST_DEVELOPMENT_EVIDENCE",
        "dataset": "MMRDR-UWF",
        "split": "official ts*; described as patient-level in dataset paper",
        "threshold": threshold,
        "threshold_source": "development validation only, frozen before test scoring",
        "metrics": result,
        "specificity_wilson_95ci": list(specificity_ci),
        "missed_referable_one_sided_95_ucb": missed_risk_ucb,
        "passes_75_specificity_zero_observed_misses": bool(result["false_negative_clearance"] == 0 and result["specificity"] >= 0.75),
        "binary_action_coverage": 1.0,
        "patient_ids_in_exported_table": False,
        "patient_level_split_basis": "official dataset description; no patient IDs are exposed in the released table",
        "clinical_certification": False,
        "notes": [
            "A single held-out dataset result does not establish clinical readiness or broad external validity.",
            "The test result must not be used to retune this checkpoint or threshold.",
            "MMRDR-UWF is not paired-view acquisition evidence.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
