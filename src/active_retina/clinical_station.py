"""Local browser station for clinician-reviewed ActiveRetina-RC cases."""
from __future__ import annotations

import argparse
import cgi
import html
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import joblib

from .clinical_case_runner import (
    build_encoder,
    classify,
    classify_forced_binary,
    embed,
    enrich_decision,
    html_report,
    sha256_file,
)
from .train_clinical_artifact import ARTIFACT_VERSION


CASE_DIR = Path("clinical_station_cases")
DEFAULT_ARTIFACT = Path("models/active_retina_rc_clinical_artifact.joblib")
MODEL_STATUS = "CLINICIAN_REVIEWED_ACQUISITION_ASSISTANT_AUTONOMOUS_DIAGNOSIS_DISABLED"


def page(body: str) -> bytes:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ActiveRetina Scan</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f5f7f8; color: #17252c; }}
    header {{ background: #173b3d; color: white; padding: 16px 24px; }}
    header h1 {{ margin: 0; font-size: 22px; }}
    main {{ max-width: 680px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #d5dfe0; border-radius: 6px; padding: 22px; }}
    label {{ display: block; font-weight: 650; margin-top: 16px; }}
    input, select {{ width: 100%; box-sizing: border-box; margin-top: 7px; padding: 11px; border: 1px solid #aebcbe; border-radius: 4px; }}
    button {{ margin-top: 20px; background: #087e78; color: white; border: 0; border-radius: 4px; padding: 12px 18px; font-weight: 700; cursor: pointer; }}
    .notice {{ color: #475a60; font-size: 14px; line-height: 1.5; }}
    .result {{ font-size: 24px; font-weight: 750; line-height: 1.25; }}
    .next {{ border-top: 1px solid #d5dfe0; margin-top: 18px; padding-top: 16px; }}
    .alert {{ color: #963b22; font-weight: 700; }}
    a {{ color: #075e86; }}
  </style>
</head>
<body>
<header>
  <h1>ActiveRetina</h1>
</header>
<main>{body}</main>
</body>
</html>""".encode("utf-8")


def form_page() -> bytes:
    return page(
        """
<section>
  <h2>Retinal scan</h2>
  <p class="notice">Upload a retinal image to receive a screening assessment. A clinician must confirm the result.</p>
  <form method="post" action="/run" enctype="multipart/form-data">
    <label>Retinal image</label>
    <input type="file" name="first_image" accept="image/*" required>
    <label>Second retinal view (optional)</label>
    <input type="file" name="second_image" accept="image/*">
    <label>Image quality</label>
    <select name="quality_ok">
      <option value="yes">Image is clear and the retina is visible</option>
      <option value="no">Image is blurry, dark, or incomplete</option>
    </select>
    <label><input type="checkbox" name="forced_binary" value="yes" style="width:auto"> Always return CLEAR or REFER (unvalidated research mode)</label>
    <button type="submit">Analyze scan</button>
    <p class="notice alert">Research software only. The binary mode is not clinically validated and must not guide patient care.</p>
  </form>
</section>
"""
    )


def assessment_summary(decision: dict[str, Any]) -> dict[str, str]:
    """Translate internal policy labels to a concise, cautious scan result."""
    recommendation = decision.get("recommendation", "")
    if recommendation == "manual_review_image_quality_failed":
        return {"result": "Image quality is not sufficient", "next": "Capture a clearer retinal image."}
    if recommendation == "second_view_required_before_recommendation":
        return {"result": "A second retinal image is needed", "next": "Capture the other retinal view, then review again."}
    if recommendation in {"manual_review_likely_referable", "manual_review_likely_referable_after_pair"}:
        return {"result": "Possible referable diabetic retinopathy", "next": "Arrange clinician assessment."}
    if recommendation in {"manual_review_low_referable_risk", "manual_review_low_referable_risk_after_pair"}:
        return {"result": "Lower likelihood of referable diabetic retinopathy", "next": "A clinician must confirm this assessment."}
    return {"result": "Clinician review needed", "next": "Do not make a screening decision from this result alone."}


def public_scan_response(result: dict[str, Any], report_url: str) -> dict[str, Any]:
    """Return only the concise user-facing scan result; audit data stays stored."""
    decision = result["decision"]
    if decision.get("binary_mode"):
        assessment = {
            "result": f"Research-model output: {decision['action']}",
            "next": "This output is unvalidated; do not use it for patient-care decisions.",
        }
    else:
        assessment = assessment_summary(decision)
    response = {
        "case_id": result["case_id"],
        "assessment": assessment["result"],
        "next_step": assessment["next"],
        "clinician_confirmation_required": True,
        "autonomous_diagnosis": False,
        "report_url": report_url,
    }
    if decision.get("binary_mode"):
        response["binary_output"] = decision["action"]
        response["binary_model_status"] = decision["binary_mode"]
        response["binary_threshold"] = decision["binary_threshold"]
        response["is_7013_policy"] = False
    return response


def get_field_text(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    item = form[name] if name in form else None
    if item is None or isinstance(item, list):
        return default
    return str(item.value or default)


def save_upload(form: cgi.FieldStorage, name: str, case_dir: Path) -> Path | None:
    item = form[name] if name in form else None
    if item is None or isinstance(item, list) or not item.filename:
        return None
    suffix = Path(item.filename).suffix.lower() or ".jpg"
    out = case_dir / f"{name}{suffix}"
    with out.open("wb") as f:
        data = item.file.read()
        f.write(data)
    return out


def sanitize_case_id(raw_case_id: str) -> str:
    return "".join(ch for ch in raw_case_id if ch.isalnum() or ch in "-_")[:80] or "case"


def analyze_case(
    artifact: dict[str, Any],
    model: Any,
    transform: Any,
    case_id: str,
    operator_id: str,
    first_path: Path,
    second_path: Path | None,
    quality_ok: bool,
    forced_binary: bool = False,
) -> dict[str, Any]:
    first = embed(first_path, model, transform)
    second = embed(second_path, model, transform) if second_path else None
    if forced_binary:
        decision = classify_forced_binary(artifact, first, second, quality_ok)
    else:
        decision = enrich_decision(classify(artifact, first, second))
    if not quality_ok and not forced_binary:
        decision = {
            **decision,
            "action": "DEFER_TO_CLINICIAN",
            "recommendation": "manual_review_image_quality_failed",
            "needs_second_view": False,
            "recommended_next_step": "repeat_or_review_image_quality_before_interpretation",
        }
    return {
        "case_id": case_id,
        "operator_id": operator_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": artifact.get("version", ARTIFACT_VERSION),
        "model_status": MODEL_STATUS,
        "feature_source": "live_resnet50_encoder",
        "inputs": {
            "first_image_path": str(first_path),
            "second_image_path": str(second_path) if second_path else None,
            "first_image_sha256": sha256_file(first_path),
            "second_image_sha256": sha256_file(second_path) if second_path else None,
            "quality_ok": quality_ok,
        },
        "decision": decision,
        "manual_review_required": True,
        "autonomous_diagnosis": False,
        "binary_output_mode": forced_binary,
        "binary_output_validated": False if forced_binary else None,
    }


def persist_case(case_id: str, result: dict[str, Any]) -> Path:
    case_dir = CASE_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (case_dir / "report.html").write_text(html_report(result), encoding="utf-8")
    return case_dir


class StationHandler(BaseHTTPRequestHandler):
    artifact: dict[str, Any]
    model: Any
    transform: Any

    def write(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.write(200, form_page())
            return
        if parsed.path == "/case":
            case_id = parse_qs(parsed.query).get("id", [""])[0]
            report = CASE_DIR / case_id / "report.html"
            if report.exists():
                self.write(200, report.read_bytes())
            else:
                self.write(404, page("<section><p>Case report not found.</p></section>"))
            return
        self.write(404, page("<section><p>Not found.</p></section>"))

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/analyze":
            self.api_analyze()
            return
        if parsed_path != "/run":
            self.write(404, page("<section><p>Not found.</p></section>"))
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
        )
        raw_case_id = get_field_text(form, "case_id", f"case-{datetime.now(timezone.utc).timestamp()}")
        case_id = sanitize_case_id(raw_case_id)
        operator_id = get_field_text(form, "operator_id", "local_station")
        quality_ok = get_field_text(form, "quality_ok", "yes") == "yes"
        forced_binary = get_field_text(form, "forced_binary", "no") == "yes"
        case_dir = CASE_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        first_path = save_upload(form, "first_image", case_dir)
        second_path = save_upload(form, "second_image", case_dir)
        if first_path is None:
            self.write(400, page("<section><p>First image is required.</p></section>"))
            return

        result = analyze_case(
            self.artifact, self.model, self.transform, case_id, operator_id,
            first_path, second_path, quality_ok, forced_binary,
        )
        persist_case(case_id, result)
        if forced_binary:
            assessment = {
                "result": f"Research-model output: {result['decision']['action']}",
                "next": "This output is unvalidated; do not use it for patient-care decisions.",
            }
        else:
            assessment = assessment_summary(result["decision"])
        self.write(
            200,
            page(
                f"""
<section>
  <h2>Scan result</h2>
  <p class="result">{html.escape(assessment['result'])}</p>
  <p class="next">{html.escape(assessment['next'])}</p>
  <p class="notice alert">{ 'Forced binary research output; not clinically validated and not for patient-care decisions.' if forced_binary else 'This is a model assessment, not a diagnosis. Clinician confirmation is required.' }</p>
  <p><a href="/case?id={html.escape(case_id)}">View detailed report</a></p>
  <p><a href="/">Scan another image</a></p>
</section>
"""
            ),
        )

    def api_analyze(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            case_id = sanitize_case_id(str(payload.get("case_id") or f"api-{datetime.now(timezone.utc).timestamp()}"))
            first_path = Path(str(payload["first_image_path"]))
            second_raw = payload.get("second_image_path")
            second_path = Path(str(second_raw)) if second_raw else None
            if not first_path.exists() or (second_path is not None and not second_path.exists()):
                raise FileNotFoundError("Input image path does not exist.")
            result = analyze_case(
                self.artifact,
                self.model,
                self.transform,
                case_id,
                str(payload.get("operator_id") or "api"),
                first_path,
                second_path,
                bool(payload.get("quality_ok", True)),
                str(payload.get("mode", "")) == "forced_binary_research",
            )
            case_dir = persist_case(case_id, result)
            response = public_scan_response(result, f"/case?id={case_id}")
            self.write(200, json.dumps(response, indent=2, sort_keys=True).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as exc:
            body = {"error": type(exc).__name__, "message": str(exc)}
            self.write(400, json.dumps(body, indent=2, sort_keys=True).encode("utf-8"), "application/json; charset=utf-8")


def run(host: str, port: int, artifact_path: Path) -> None:
    StationHandler.artifact = joblib.load(artifact_path)
    StationHandler.model, StationHandler.transform = build_encoder()
    server = ThreadingHTTPServer((host, port), StationHandler)
    print(f"ActiveRetina-RC station running at http://{host}:{port}")
    server.serve_forever()


def smoke() -> None:
    artifact = joblib.load(DEFAULT_ARTIFACT)
    assert artifact.get("version")
    form = form_page().decode("utf-8")
    assert "Analyze scan" in form
    assert 'name="forced_binary"' in form
    assert "70.13" not in form and "63.12" not in form
    assert assessment_summary({"recommendation": "manual_review_likely_referable"})["result"] == "Possible referable diabetic retinopathy"
    print("clinical station smoke passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    run(args.host, args.port, Path(args.artifact))


if __name__ == "__main__":
    main()
