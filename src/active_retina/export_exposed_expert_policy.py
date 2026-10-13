"""Freeze the exposed 70.13% recipe as a fixed-score replay artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from active_retina.audit_exposed_pooled_specificity import (
    ROOT,
    _align,
    _binary_metrics,
    _deepdrid_predictions,
    _ddr_predictions,
    _drtid_predictions,
    _read,
    build_report,
)
from active_retina.exposed_expert_policy import score_case


def _references(frame: pd.DataFrame, **columns: pd.Series | np.ndarray) -> dict[str, list[float]]:
    references = {}
    for name, values in columns.items():
        arr = np.asarray(values, dtype=float)
        if len(arr) != len(frame) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid calibration reference column: {name}")
        references[name] = np.sort(arr).tolist()
    return references


def export(results: Path, output: Path) -> dict[str, Any]:
    report = build_report(results)
    ddr, _, _ = _ddr_predictions(results)
    deep, _ = _deepdrid_predictions(results)
    drtid, _, _, _ = _drtid_predictions(results)
    ddr = ddr.sort_values("eye_id").reset_index(drop=True)
    deep = deep.sort_values("eye_id").reset_index(drop=True)
    drtid = drtid.sort_values("eye_id").reset_index(drop=True)

    ddr_ord = _align(results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv",
                     ddr["eye_id"], ["max_grade_margin"])["max_grade_margin"]
    ddr_margin = _align(results / "max_grade_margin_audit_20261011/validation_predictions.csv",
                        ddr["eye_id"], ["max_grade_margin"])["max_grade_margin"]

    deep_quality = _align(results / "rtx_ordinal_attention_residual_20261011/validation_predictions.csv",
                          deep["eye_id"], ["quality"])["quality"]
    deep_margin = _align(results / "rtx_convnext_sourcebalanced_20261011/validation_predictions.csv",
                         deep["eye_id"], ["max_grade_margin"])["max_grade_margin"]

    rankheavy = _align(results / "rtx_drtid_rankheavy_viewaux_flipaug_current_split/pooled_validation_frozen/predictions.csv",
                       drtid["eye_id"], ["paired"])["paired"]
    drtid_margin_frame = _read(results / "rtx_fundus_b5_all_sources_20261011/validation_predictions.csv")
    drtid_margin_frame = drtid_margin_frame[
        (drtid_margin_frame["split"] == "valid") & (drtid_margin_frame["source_dataset"] == "drtid_paired")
    ]
    drtid_margin = _align(drtid_margin_frame, drtid["eye_id"], ["max_grade_margin"])["max_grade_margin"]

    source_metrics = report["source_results"]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_status": "EXPOSED_REPLAY_ONLY_NOT_CLINICALLY_VALIDATED",
        "intended_use": "Deterministic replay of the inspected 70.13% development recipe from component model scores.",
        "prohibited_interpretation": "Not an independent evaluation, calibrated clinical model, or evidence of future-case specificity.",
        "calibration_origin": "The score-reference distributions and zero-miss cutoffs were derived from the same repeatedly inspected validation cohort used to report 70.13%.",
        "sources": {
            "DDR": {
                "risk_threshold": source_metrics["DDR"]["threshold"],
                "quality_threshold": 0.0018598887836560607,
                "references": _references(ddr,
                    aptos_margin=ddr["max_grade_margin"],
                    ordinal_attention_margin=ddr_ord,
                    margin_audit_margin=ddr_margin),
            },
            "DeepDRiD": {
                "risk_threshold": source_metrics["DeepDRiD"]["threshold"],
                "references": _references(deep,
                    ordinal_quality=deep_quality,
                    sourcebalanced_margin=deep_margin,
                    fundus_risk=deep["risk"]),
            },
            "DRTiD": {
                "risk_threshold": source_metrics["DRTiD_mapping_invariant_fusion"]["threshold"],
                "references": _references(drtid,
                    rankheavy_pair=rankheavy,
                    fundus_margin=drtid_margin,
                    geometry_pair=drtid["paired"]),
            },
        },
    }

    prediction_rows: list[dict[str, Any]] = []
    for source, frame in (("DDR", ddr), ("DeepDRiD", deep), ("DRTiD", drtid)):
        for index, row in frame.iterrows():
            if source == "DDR":
                features = {
                    "aptos_margin": float(row["max_grade_margin"]),
                    "ordinal_attention_margin": float(ddr_ord.iloc[index]),
                    "margin_audit_margin": float(ddr_margin.iloc[index]),
                    "quality": float(row["quality"]),
                }
                grade = int(row["referable"])
                ungradable = int(row["ungradable"])
            elif source == "DeepDRiD":
                features = {
                    "ordinal_quality": float(deep_quality.iloc[index]),
                    "sourcebalanced_margin": float(deep_margin.iloc[index]),
                    "fundus_risk": float(row["risk"]),
                }
                grade = int(row["referable"])
                ungradable = 0
            else:
                features = {
                    "rankheavy_pair": float(rankheavy.iloc[index]),
                    "fundus_margin": float(drtid_margin.iloc[index]),
                    "geometry_pair": float(row["paired"]),
                }
                grade = int(row["referable_dr"])
                ungradable = 0
            decision = score_case(artifact, source, features)
            prediction_rows.append({
                "source": source, "eye_id": row["eye_id"], "grade": int(row["grade"]),
                "referable": grade, "ungradable": ungradable, **features, **decision,
            })

    pred = pd.DataFrame(prediction_rows)
    per_source: dict[str, dict[str, Any]] = {}
    for source, group in pred.groupby("source", sort=False):
        labels = group["referable"].to_numpy(dtype=bool)
        ungradable = group["ungradable"].to_numpy(dtype=bool)
        refer = group["action"].eq("REFER").to_numpy()
        negatives = ~labels & ~ungradable
        per_source[source] = {
            "cases": int(len(group)),
            "gradable_nonreferable": int(negatives.sum()),
            "true_negatives": int((negatives & ~refer).sum()),
            "missed_referable": int((labels & ~refer).sum()),
            "missed_ungradable": int((ungradable & ~refer).sum()),
            "action_coverage": float(group["action"].isin(["REFER", "CLEAR"]).mean()),
        }
    metrics = {
        "n_cases": int(len(pred)),
        "n_gradable_nonreferable": sum(item["gradable_nonreferable"] for item in per_source.values()),
        "true_negatives": sum(item["true_negatives"] for item in per_source.values()),
        "missed_referable": sum(item["missed_referable"] for item in per_source.values()),
        "missed_ungradable": sum(item["missed_ungradable"] for item in per_source.values()),
        "action_coverage": min(item["action_coverage"] for item in per_source.values()),
    }
    metrics["specificity"] = metrics["true_negatives"] / metrics["n_gradable_nonreferable"]
    metrics["per_source"] = per_source
    artifact["same_cohort_replay_metrics"] = metrics
    artifact["replay_matches_reference"] = (
        metrics["true_negatives"] == 270
        and metrics["missed_referable"] == 0
        and metrics["missed_ungradable"] == 0
        and metrics["action_coverage"] == 1.0
        and per_source["DDR"]["true_negatives"] == 134
        and per_source["DeepDRiD"]["true_negatives"] == 51
        and per_source["DRTiD"]["true_negatives"] == 85
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    pred.to_csv(output / "validation_replay.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "artifact_status": artifact["artifact_status"],
        "replay_matches_reference": artifact["replay_matches_reference"],
        "same_cohort_replay_metrics": metrics,
        "important": artifact["prohibited_interpretation"],
    }, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/deployable_exposed_baseline_7013_20261013")
    args = parser.parse_args()
    result = export(args.results_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "replay_matches_reference": result["replay_matches_reference"],
                      "metrics": result["same_cohort_replay_metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
