"""Registry for the two inspected zero-miss development baselines.

These profiles are research artifacts, not clinically validated models. The
70.13 profile replays an exposed score recipe; the 63.12 profile is an
evaluation-only post-hoc frontier and intentionally has no inference path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from active_retina.exposed_expert_policy import score_case


ROOT = Path(__file__).resolve().parents[2]
PROFILES: dict[str, dict[str, Any]] = {
    "70.13-replay": {
        "profile_id": "70.13-replay",
        "name": "Exposed pooled expert-score replay",
        "specificity_percent": 70.12987012987013,
        "true_negatives": 270,
        "negative_cases": 385,
        "observed_misses": 0,
        "action_coverage_percent": 100.0,
        "status": "EXPOSED_REPLAY_ONLY_NOT_CLINICALLY_VALIDATED",
        "inference_mode": "precomputed_component_scores_only",
        "policy_path": "results/deployable_exposed_baseline_7013_20261013_with_components/policy.json",
    },
    "63.12-posthoc": {
        "profile_id": "63.12-posthoc",
        "name": "Post-hoc zero-miss margin frontier",
        "specificity_percent": 63.116883116883116,
        "true_negatives": 243,
        "negative_cases": 385,
        "observed_misses": 0,
        "action_coverage_percent": 100.0,
        "status": "POSTHOC_DIAGNOSTIC_ONLY_NOT_A_VALID_CROSS_VALIDATED_CANDIDATE",
        "inference_mode": "evaluation_only",
        "warning": "Margins were chosen using held-out fold labels. This is not a frozen policy and cannot score new cases.",
        "diagnostic_path": "results/posthoc_zero_miss_margin_sweep_20261013.json",
    },
}


def list_profiles() -> list[dict[str, Any]]:
    """Return profile metadata without implying either result is validated."""
    return [dict(profile) for profile in PROFILES.values()]


def score_profile_case(
    profile_id: str,
    source: str,
    features: Mapping[str, float],
    policy_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the 70.13 score recipe; fail closed for diagnostic-only 63.12."""
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown baseline profile: {profile_id}")
    profile = PROFILES[profile_id]
    if profile["inference_mode"] != "precomputed_component_scores_only":
        raise ValueError(
            f"{profile_id} is evaluation-only; its held-out-label-tuned margins "
            "cannot be used to score new cases"
        )
    if policy_artifact is None:
        raise ValueError("The 70.13 replay requires its explicit policy artifact")
    result = score_case(policy_artifact, source, features)
    return {**result, "profile_id": profile_id, "profile_status": profile["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered profiles")
    parser.add_argument("--profile", choices=PROFILES, help="Select a profile")
    parser.add_argument("--source", choices=("DDR", "DeepDRiD", "DRTiD"))
    parser.add_argument("--features-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    args = parser.parse_args()

    if args.list or not args.profile:
        print(json.dumps(list_profiles(), indent=2))
        return
    if args.profile == "63.12-posthoc":
        raise SystemExit(
            "63.12-posthoc is evaluation-only; use its stored diagnostic report, "
            "not per-case inference."
        )
    if not args.source or not args.features_json or not args.policy_json:
        parser.error("70.13-replay requires --source, --features-json, and --policy-json")
    features = json.loads(args.features_json.read_text(encoding="utf-8"))
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    print(json.dumps(score_profile_case(args.profile, args.source, features, policy), indent=2))


if __name__ == "__main__":
    main()
