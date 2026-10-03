"""Run a clinical-style ActiveRetina-RC case from image paths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image

from .train_clinical_artifact import ARTIFACT_VERSION, pooled


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_encoder():
    import timm

    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    model.eval()
    return model, transform


def load_cached_features(manifest_path: Path, feature_path: Path) -> dict[str, np.ndarray]:
    """Map normalized image paths to frozen feature vectors."""
    if not manifest_path.exists() or not feature_path.exists():
        return {}
    features = np.load(feature_path, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["image_id"]
            if image_id not in features:
                continue
            path = str(Path(row["image_path"]))
            out[path] = features[image_id].astype(np.float32)
            out[str(Path(path).resolve())] = features[image_id].astype(np.float32)
    return out


def embed(path: Path, model: Any, transform: Any) -> np.ndarray:
    with Image.open(path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        return model(tensor).detach().cpu().float().numpy()[0].astype(np.float32)


def classify(artifact: dict[str, Any], first: np.ndarray, second: np.ndarray | None) -> dict[str, Any]:
    policy = artifact["policy"]
    first_x = pooled([first]).reshape(1, -1)
    first_score = float(artifact["models"]["first_binary"].predict_proba(first_x)[0, 1])
    first_grade = int(artifact["models"]["first_grade"].predict(first_x)[0])
    if first_score <= policy["stop_low"]:
        return {
            "action": "STOP_AFTER_FIRST_VIEW",
            "recommendation": "manual_review_low_referable_risk",
            "images_used": 1,
            "referable_score": first_score,
            "grade_prediction": first_grade,
            "needs_second_view": False,
        }
    if first_score >= policy["stop_high"]:
        return {
            "action": "STOP_AFTER_FIRST_VIEW",
            "recommendation": "manual_review_likely_referable",
            "images_used": 1,
            "referable_score": first_score,
            "grade_prediction": first_grade,
            "needs_second_view": False,
        }
    if second is None:
        return {
            "action": "ACQUIRE_SECOND_VIEW",
            "recommendation": "second_view_required_before_recommendation",
            "images_used": 1,
            "referable_score": first_score,
            "grade_prediction": first_grade,
            "needs_second_view": True,
        }
    pair_x = pooled([first, second]).reshape(1, -1)
    pair_score = float(artifact["models"]["paired_binary"].predict_proba(pair_x)[0, 1])
    pair_grade = int(artifact["models"]["paired_grade"].predict(pair_x)[0])
    if pair_score <= policy["second_low"]:
        action = "ACQUIRE_THEN_STOP"
        recommendation = "manual_review_low_referable_risk_after_pair"
    elif pair_score >= policy["second_high"]:
        action = "ACQUIRE_THEN_REFER"
        recommendation = "manual_review_likely_referable_after_pair"
    else:
        action = "DEFER_TO_CLINICIAN"
        recommendation = "manual_review_uncertain_after_pair"
    return {
        "action": action,
        "recommendation": recommendation,
        "images_used": 2,
        "referable_score": pair_score,
        "grade_prediction": pair_grade,
        "needs_second_view": False,
    }


def enrich_decision(decision: dict[str, Any]) -> dict[str, Any]:
    score = float(decision["referable_score"])
    grade = int(decision["grade_prediction"])
    if score < 0.15:
        impression = "low_referable_dr_probability"
    elif score >= 0.50:
        impression = "elevated_referable_dr_probability"
    else:
        impression = "indeterminate_referable_dr_probability"
    if grade <= 1:
        grade_text = "nonreferable_or_mild_dr_predicted"
    elif grade == 2:
        grade_text = "moderate_dr_predicted"
    elif grade == 3:
        grade_text = "severe_dr_predicted"
    else:
        grade_text = "proliferative_dr_predicted"
    if decision["needs_second_view"]:
        next_step = "acquire_second_retinal_view"
    elif decision["action"] == "DEFER_TO_CLINICIAN":
        next_step = "clinician_review_required_before_screening_decision"
    else:
        next_step = "clinician_review_required_before_final_decision"
    return {
        **decision,
        "referable_probability": score,
        "predicted_dr_grade": grade,
        "model_impression_for_clinician_review": impression,
        "grade_impression_for_clinician_review": grade_text,
        "recommended_next_step": next_step,
    }


def html_report(result: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ActiveRetina-RC Case Report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;max-width:900px}}.warn{{color:#9b2226;font-weight:bold}}dt{{font-weight:bold;margin-top:12px}}</style>
</head><body>
<h1>ActiveRetina-RC Case Report</h1>
<p class="warn">Manual review required. This software output is not autonomous diagnosis.</p>
<dl>
<dt>Case</dt><dd>{result['case_id']}</dd>
<dt>Action</dt><dd>{result['decision']['action']}</dd>
<dt>Recommendation</dt><dd>{result['decision']['recommendation']}</dd>
<dt>Model impression</dt><dd>{result['decision'].get('model_impression_for_clinician_review', 'not available')}</dd>
<dt>Referable DR score</dt><dd>{result['decision']['referable_score']:.4f}</dd>
<dt>Five-class grade prediction</dt><dd>{result['decision']['grade_prediction']}</dd>
<dt>Next step</dt><dd>{result['decision'].get('recommended_next_step', 'manual review required')}</dd>
<dt>Images used</dt><dd>{result['decision']['images_used']}</dd>
<dt>Model version</dt><dd>{result['model_version']}</dd>
<dt>Status</dt><dd>{result['model_status']}</dd>
</dl>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="models/active_retina_rc_clinical_artifact.joblib")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--first-image", required=True)
    parser.add_argument("--second-image")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html")
    parser.add_argument("--manifest", default="results/deepdrid_manifest.csv")
    parser.add_argument("--feature-cache", default="results/deepdrid_resnet50_features.npz")
    parser.add_argument("--force-live-encoder", action="store_true")
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    first_path = Path(args.first_image)
    second_path = Path(args.second_image) if args.second_image else None
    cached = load_cached_features(Path(args.manifest), Path(args.feature_cache))
    first_key = str(first_path)
    second_key = str(second_path) if second_path else None
    feature_source = "frozen_feature_cache"
    if not args.force_live_encoder and first_key in cached and (second_path is None or second_key in cached):
        first = cached[first_key]
        second = cached[second_key] if second_key else None
    else:
        feature_source = "live_resnet50_encoder"
        model, transform = build_encoder()
        first = embed(first_path, model, transform)
        second = embed(second_path, model, transform) if second_path else None
    result = {
        "case_id": args.case_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": artifact.get("version", ARTIFACT_VERSION),
        "model_status": "CLINICAL_WORKFLOW_SOFTWARE_READY_NOT_AUTONOMOUS_DIAGNOSIS",
        "feature_source": feature_source,
        "inputs": {
            "first_image_sha256": sha256_file(first_path),
            "second_image_sha256": sha256_file(second_path) if second_path else None,
        },
        "decision": enrich_decision(classify(artifact, first, second)),
        "manual_review_required": True,
        "autonomous_diagnosis": False,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_html:
        Path(args.output_html).write_text(html_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
