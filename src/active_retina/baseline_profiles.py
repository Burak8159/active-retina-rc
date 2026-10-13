"""Registry for the two inspected zero-miss development baselines.

These profiles are research artifacts, not clinically validated models. The
70.13 profile replays an exposed score recipe; the 63.12 profile is an
evaluation-only post-hoc frontier and intentionally has no inference path.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from active_retina.exposed_expert_policy import score_case


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_7013_POLICY = ROOT / "configs/research_baselines/70_13_exposed_policy.json"
DEFAULT_6312_REPORT = ROOT / "configs/research_baselines/63_12_posthoc_diagnostic.json"
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
        "policy_path": "configs/research_baselines/70_13_exposed_policy.json",
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
        "diagnostic_path": "configs/research_baselines/63_12_posthoc_diagnostic.json",
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


def diagnostic_report_6312(report_path: Path = DEFAULT_6312_REPORT) -> dict[str, Any]:
    """Load the cohort-level 63.12 diagnostic; it does not score new cases."""
    if not report_path.is_file():
        raise FileNotFoundError(f"63.12 diagnostic report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != PROFILES["63.12-posthoc"]["status"]:
        raise ValueError("63.12 report status does not match the registered diagnostic profile")
    return report


def make_handler(policy_path: Path, diagnostic_path: Path):
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: Any) -> None:
            body = json.dumps(value, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            status, result = dispatch_request("GET", self.path, {}, policy, diagnostic_path)
            self._json(status, result)

        def do_POST(self) -> None:
            if self.path != "/score":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("Request body must be 1..65536 bytes")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                status, result = dispatch_request("POST", self.path, payload, policy, diagnostic_path)
                self._json(status, result)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._json(422, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def dispatch_request(
    method: str,
    path: str,
    payload: Mapping[str, Any],
    policy: Mapping[str, Any],
    diagnostic_path: Path = DEFAULT_6312_REPORT,
) -> tuple[int, dict[str, Any] | list[dict[str, Any]]]:
    """Pure route logic, independently testable without opening a socket."""
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "use": "research_only_not_clinical"}
    if method == "GET" and path == "/profiles":
        return 200, list_profiles()
    if method == "GET" and path == "/diagnostics/63.12-posthoc":
        try:
            return 200, diagnostic_report_6312(diagnostic_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return 503, {"error": str(exc)}
    if method == "POST" and path == "/score":
        try:
            result = score_profile_case(
                str(payload.get("profile_id", "")),
                str(payload.get("source", "")),
                payload.get("features", {}),
                policy,
            )
            result["use"] = "research_replay_not_clinical_diagnosis"
            return 200, result
        except (ValueError, KeyError, TypeError) as exc:
            return 422, {"error": str(exc)}
    return 404, {"error": "not_found"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered profiles")
    parser.add_argument("--profile", choices=PROFILES, help="Select a profile")
    parser.add_argument("--source", choices=("DDR", "DeepDRiD", "DRTiD"))
    parser.add_argument("--features-json", type=Path)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_7013_POLICY)
    parser.add_argument("--diagnostic-json", type=Path, default=DEFAULT_6312_REPORT)
    parser.add_argument("--serve", action="store_true", help="Run the local research API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.serve:
        server = ThreadingHTTPServer(
            (args.host, args.port), make_handler(args.policy_json, args.diagnostic_json)
        )
        print(f"Research baseline API listening at http://{args.host}:{args.port}")
        server.serve_forever()
        return
    if args.list or not args.profile:
        print(json.dumps(list_profiles(), indent=2))
        return
    if args.profile == "63.12-posthoc":
        print(json.dumps(diagnostic_report_6312(args.diagnostic_json), indent=2))
        return
    if not args.source or not args.features_json:
        parser.error("70.13-replay requires --source and --features-json")
    features = json.loads(args.features_json.read_text(encoding="utf-8"))
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    print(json.dumps(score_profile_case(args.profile, args.source, features, policy), indent=2))


if __name__ == "__main__":
    main()
