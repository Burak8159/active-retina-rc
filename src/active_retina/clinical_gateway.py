"""Clinical-facing safety gateway for ActiveRetina-RC.

This is software-side infrastructure for a locked research/clinical-decision
support prototype. It deliberately fails closed when the model is not certified.
It does not make the current model a regulated medical device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_VERSION = "active-retina-rc-deepdrid-resnet50-cv-2026-10-03"
MODEL_STATUS = "RESEARCH_ONLY_NOT_CLINICALLY_CERTIFIED"
SUPPORTED_SCHEMA_VERSION = "1.0"
DEFAULT_LOW = 0.05
DEFAULT_HIGH = 0.85
DEFAULT_TWO_VIEW_REFERRAL = 0.50
UNCERTIFIED_REASON = (
    "Formal calibration audit does not certify a 5% missed-referable-DR risk target; "
    "use for research workflow simulation only."
)


@dataclass(frozen=True)
class GatewayInput:
    schema_version: str
    case_id: str
    operator_id: str
    first_view_score: float
    two_view_score: float | None = None
    first_view_quality_ok: bool = True
    second_view_quality_ok: bool = True
    anatomy_mapping_verified: bool = False
    intended_use: str = "research"


@dataclass(frozen=True)
class GatewayOutput:
    schema_version: str
    case_id: str
    model_version: str
    model_status: str
    timestamp_utc: str
    action: str
    recommendation: str
    images_required: int
    risk_score_used: float | None
    must_review: bool
    clinically_actionable: bool
    reasons: list[str]
    input_sha256: str


def sha256_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_input(path: Path) -> GatewayInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GatewayInput(**payload)


def validate(inp: GatewayInput) -> list[str]:
    reasons: list[str] = []
    if inp.schema_version != SUPPORTED_SCHEMA_VERSION:
        reasons.append(f"Unsupported schema_version={inp.schema_version}")
    if inp.intended_use != "research":
        reasons.append("Clinical intended_use is blocked because this model is not certified.")
    if not (0.0 <= inp.first_view_score <= 1.0):
        reasons.append("first_view_score must be in [0, 1].")
    if inp.two_view_score is not None and not (0.0 <= inp.two_view_score <= 1.0):
        reasons.append("two_view_score must be in [0, 1].")
    if not inp.first_view_quality_ok:
        reasons.append("First-view image quality failed.")
    if inp.two_view_score is not None and not inp.second_view_quality_ok:
        reasons.append("Second-view image quality failed.")
    if not inp.anatomy_mapping_verified:
        reasons.append("Anatomical field mapping is not verified for this case/workflow.")
    reasons.append(UNCERTIFIED_REASON)
    return reasons


def decide(inp: GatewayInput) -> GatewayOutput:
    payload = asdict(inp)
    reasons = validate(inp)
    hard_block = any(
        reason.startswith("Unsupported")
        or reason.startswith("Clinical intended_use")
        or reason.endswith("failed.")
        or reason.startswith("Anatomical")
        for reason in reasons
    )

    action = "DEFER_TO_CLINICIAN"
    recommendation = "manual_review_required"
    images_required = 1 if inp.two_view_score is None else 2
    score_used: float | None = inp.first_view_score

    if not hard_block and inp.intended_use == "research":
        if inp.first_view_score <= DEFAULT_LOW:
            action = "STOP_AFTER_FIRST_VIEW"
            recommendation = "research_negative_low_risk"
            images_required = 1
            score_used = inp.first_view_score
        elif inp.first_view_score >= DEFAULT_HIGH:
            action = "STOP_AFTER_FIRST_VIEW"
            recommendation = "research_positive_referable"
            images_required = 1
            score_used = inp.first_view_score
        elif inp.two_view_score is None:
            action = "ACQUIRE_SECOND_VIEW"
            recommendation = "research_second_view_needed"
            images_required = 2
            score_used = inp.first_view_score
        elif inp.two_view_score >= DEFAULT_TWO_VIEW_REFERRAL:
            action = "ACQUIRE_THEN_REFER"
            recommendation = "research_positive_referable_after_second_view"
            images_required = 2
            score_used = inp.two_view_score
        else:
            action = "DEFER_TO_CLINICIAN"
            recommendation = "research_uncertain_manual_review"
            images_required = 2
            score_used = inp.two_view_score

    return GatewayOutput(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        case_id=inp.case_id,
        model_version=MODEL_VERSION,
        model_status=MODEL_STATUS,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        action=action,
        recommendation=recommendation,
        images_required=images_required,
        risk_score_used=score_used,
        must_review=True,
        clinically_actionable=False,
        reasons=reasons,
        input_sha256=sha256_payload(payload),
    )


def append_audit(path: Path, output: GatewayOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(output), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-log", default="results/clinical_gateway_audit.jsonl")
    args = parser.parse_args()
    out = decide(load_input(Path(args.input)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(asdict(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_audit(Path(args.audit_log), out)


if __name__ == "__main__":
    main()
