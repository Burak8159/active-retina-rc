"""Run the complete untouched certification package.

This command intentionally does not tune thresholds or repair the model. It is
the one-shot path for a new cohort after the autonomous candidate is frozen.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(args, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            json.dumps(
                {
                    "status": "CERTIFICATION_PACKAGE_COMMAND_FAILED",
                    "command": args,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-4000:],
                    "stderr": proc.stderr[-4000:],
                },
                indent=2,
            )
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"stdout": proc.stdout}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-kind", choices=["canonical", "mbrset", "drtid"], required=True)
    parser.add_argument("--labels", help="mBRSET labels CSV when --cohort-kind=mbrset")
    parser.add_argument("--image-root", help="mBRSET image root when --cohort-kind=mbrset")
    parser.add_argument("--drtid-root", help="DRTiD root folder when --cohort-kind=drtid")
    parser.add_argument("--drtid-include-training", action="store_true")
    parser.add_argument("--drtid-include-testing", action="store_true")
    parser.add_argument(
        "--suffix-view-map",
        help="Audited mBRSET filename suffix anatomy map, e.g. 1:macula,2:optic_disc,3:macula,4:optic_disc.",
    )
    parser.add_argument("--canonical-input", help="Raw canonical paired-eye CSV when --cohort-kind=canonical")
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--output-dir", default="results/untouched_certification_run")
    parser.add_argument("--artifact", default="models/active_retina_rc_autonomous_candidate.joblib")
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "certification_manifest.csv"
    prep_summary = out / "manifest_preparation_summary.json"
    validation_summary = out / "untouched_validation_summary.json"
    feature_cache = out / "certification_features.npz"
    predictions = out / "certification_predictions.csv"
    certification_summary = out / "certification_summary.json"
    final_decision = out / "final_deployability_decision.json"

    py = sys.executable
    if args.cohort_kind == "mbrset":
        if not args.labels or not args.image_root:
            raise SystemExit("--labels and --image-root are required for mBRSET")
        prep_cmd = [
            py,
            "-m",
            "active_retina.prepare_mbrset_for_untouched_review",
            "--labels",
            args.labels,
            "--image-root",
            args.image_root,
            "--output-manifest",
            str(manifest),
            "--output-summary",
            str(prep_summary),
        ]
        if args.suffix_view_map:
            prep_cmd.extend(["--suffix-view-map", args.suffix_view_map])
        prep = run(prep_cmd)
    elif args.cohort_kind == "drtid":
        if not args.drtid_root:
            raise SystemExit("--drtid-root is required for DRTiD")
        prep_cmd = [
            py,
            "-m",
            "active_retina.prepare_drtid_for_untouched_review",
            "--drtid-root",
            args.drtid_root,
            "--output-manifest",
            str(manifest),
            "--output-summary",
            str(prep_summary),
        ]
        if args.drtid_include_training:
            prep_cmd.append("--include-training")
        if args.drtid_include_testing:
            prep_cmd.append("--include-testing")
        prep = run(prep_cmd)
    else:
        if not args.canonical_input:
            raise SystemExit("--canonical-input is required for canonical cohorts")
        prep = run(
            [
                py,
                "-m",
                "active_retina.prepare_untouched_certification_cohort",
                "--input",
                args.canonical_input,
                "--output-manifest",
                str(manifest),
                "--output-summary",
                str(prep_summary),
                "--source-name",
                "untouched_external_certification",
            ]
        )

    if prep.get("status") not in {
        "MBRSET_PAIRED_MANIFEST_CREATED",
        "DRTID_CERTIFICATION_MANIFEST_PREPARED",
        "UNTOUCHED_CERTIFICATION_MANIFEST_PREPARED",
    }:
        decision = {
            "status": "NO_GO_CERTIFICATION_MANIFEST_NOT_CREATED",
            "manifest_preparation": prep,
            "next_action": "Provide metadata with complete same-eye macula and optic-disc pairs.",
        }
        write_json(final_decision, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return

    validation = run(
        [
            py,
            "-m",
            "active_retina.validate_untouched_cohort",
            "--manifest",
            str(manifest),
            "--attestation",
            args.attestation,
            "--output",
            str(validation_summary),
        ]
    )
    if validation.get("status") != "UNTOUCHED_CERTIFICATION_ELIGIBLE":
        decision = {
            "status": "NO_GO_UNTOUCHED_VALIDATION_FAILED",
            "manifest_preparation": prep,
            "untouched_validation": validation,
            "next_action": "Fix the blocking items before running the frozen model.",
        }
        write_json(final_decision, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return

    certification = run(
        [
            py,
            "-m",
            "active_retina.certify_autonomous_candidate",
            "--manifest",
            str(manifest),
            "--artifact",
            args.artifact,
            "--feature-cache",
            str(feature_cache),
            "--output-predictions",
            str(predictions),
            "--output-summary",
            str(certification_summary),
            "--model-name",
            args.model_name,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
    )
    go = certification.get("decision") == "GO_BINARY_80_0_100_RESEARCH_TARGET_MET"
    decision = {
        "status": "GO_RETROSPECTIVE_BINARY_80_0_100_MET_NOT_CLINICAL_VALIDATION" if go else "NO_GO_AUTONOMOUS_CERTIFICATION_FAILED",
        "manifest_preparation": prep,
        "untouched_validation": validation,
        "certification": certification,
        "product_claim": (
            "The frozen model met the prespecified retrospective 80/0/100 computational target on this untouched cohort; this is not clinical validation."
            if go
            else "Frozen computation ran correctly, but autonomous diagnosis is not certified."
        ),
        "paper_claim": (
            "Report the locked retrospective result and its uncertainty; do not imply prospective clinical validation."
            if go
            else "Report as failed untouched certification and retain clinician-reviewed/conditional claim."
        ),
    }
    write_json(final_decision, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
