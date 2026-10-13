"""Inference-only evaluator for the frozen exposed 70.13% score recipe.

The bundled score references and cutoffs are derived from inspected validation
patients. This module makes the old calculation executable per case; it does
not turn its result into independent, clinically deployable evidence.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


FEATURES = {
    "DDR": ("aptos_margin", "ordinal_attention_margin", "margin_audit_margin"),
    "DeepDRiD": ("ordinal_quality", "sourcebalanced_margin", "fundus_risk"),
    "DRTiD": ("rankheavy_pair", "fundus_margin", "geometry_pair"),
}


def _percentile(value: float, sorted_reference: list[float]) -> float:
    if not np.isfinite(value) or not sorted_reference:
        raise ValueError("A finite score and nonempty reference distribution are required")
    reference = np.asarray(sorted_reference, dtype=float)
    if not np.isfinite(reference).all() or np.any(reference[1:] < reference[:-1]):
        raise ValueError("Reference scores must be finite and sorted")
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return ((left + 1) + right) / (2.0 * len(reference))


def score_case(
    artifact: Mapping[str, Any], source: str, features: Mapping[str, float]
) -> dict[str, float | str | bool]:
    """Score one case from precomputed expert outputs and emit a binary action."""
    if artifact.get("artifact_status") != "EXPOSED_REPLAY_ONLY_NOT_CLINICALLY_VALIDATED":
        raise ValueError("Unexpected artifact status; refusing to apply an unrecognized policy")
    if source not in FEATURES or source not in artifact["sources"]:
        raise ValueError(f"Unsupported source: {source}")
    spec = artifact["sources"][source]
    names = FEATURES[source]
    required = names + (("quality",) if source == "DDR" else ())
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"Missing required model outputs: {missing}")
    ranks = [_percentile(float(features[name]), spec["references"][name]) for name in names]
    if source == "DDR":
        risk = 0.5 * ranks[0] + 0.1 * ranks[1] + 0.4 * ranks[2]
        forced = float(features["quality"]) >= float(spec["quality_threshold"])
    elif source == "DeepDRiD":
        risk = 0.2 * ranks[0] + 0.4 * ranks[1] + 0.4 * ranks[2]
        forced = False
    else:
        risk = 0.8 * (0.9 * ranks[0] + 0.1 * ranks[1]) + 0.2 * ranks[2]
        forced = False
    refer = forced or risk >= float(spec["risk_threshold"])
    return {
        "source": source,
        "risk_score": float(risk),
        "risk_threshold": float(spec["risk_threshold"]),
        "quality_routed": bool(forced),
        "action": "REFER" if refer else "CLEAR",
    }
