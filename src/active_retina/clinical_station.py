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

from .clinical_case_runner import build_encoder, classify, embed, enrich_decision, html_report, sha256_file
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
  <title>ActiveRetina-RC Clinical Station</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #15202b; }}
    header {{ background: #102a43; color: white; padding: 18px 28px; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    label {{ display: block; font-weight: 650; margin-top: 14px; }}
    input, select {{ width: 100%; box-sizing: border-box; margin-top: 6px; padding: 10px; border: 1px solid #b8c2cc; border-radius: 6px; }}
    button {{ margin-top: 18px; background: #0b6bcb; color: white; border: 0; border-radius: 6px; padding: 11px 16px; font-weight: 700; cursor: pointer; }}
    .warn {{ color: #9b2226; font-weight: 800; }}
    .ok {{ color: #0f766e; font-weight: 800; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; text-align: left; padding: 9px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>ActiveRetina-RC Clinical Station</h1>
  <p>Clinician-reviewed acquisition assistant. Autonomous diagnosis is disabled.</p>
</header>
<main>{body}</main>
</body>
</html>""".encode("utf-8")


def form_page() -> bytes:
    return page(
        """
<section>
  <p class="warn">Manual clinician review is required for every case. This station supports image acquisition workflow and reporting; it does not independently diagnose.</p>
  <form method="post" action="/run" enctype="multipart/form-data">
    <label>Case ID</label>
    <input name="case_id" required placeholder="CASE-0001">
    <label>Operator ID</label>
    <input name="operator_id" required placeholder="operator initials or station login">
    <label>First fundus image</label>
    <input type="file" name="first_image" accept="image/*" required>
    <label>Second fundus image</label>
    <input type="file" name="second_image" accept="image/*">
    <label>Image quality gate</label>
    <select name="quality_ok">
      <option value="yes">Quality checked by operator</option>
      <option value="no">Quality failed or uncertain</option>
    </select>
    <button type="submit">Run Case</button>
  </form>
</section>
<section>
  <h2>Station Rules</h2>
  <table>
    <tr><th>Rule</th><th>Behavior</th></tr>
    <tr><td>Uncertain case</td><td>Acquire second view or defer to clinician.</td></tr>
    <tr><td>Low/high score</td><td>Show recommendation, but require manual review.</td></tr>
    <tr><td>Quality failure</td><td>Defer to clinician; do not rely on model output.</td></tr>
    <tr><td>Audit</td><td>Save JSON result, report HTML, image hashes, and timestamp.</td></tr>
  </table>
</section>
"""
    )


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
) -> dict[str, Any]:
    first = embed(first_path, model, transform)
    second = embed(second_path, model, transform) if second_path else None
    decision = enrich_decision(classify(artifact, first, second))
    if not quality_ok:
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
        operator_id = get_field_text(form, "operator_id", "unknown")
        quality_ok = get_field_text(form, "quality_ok", "yes") == "yes"
        case_dir = CASE_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        first_path = save_upload(form, "first_image", case_dir)
        second_path = save_upload(form, "second_image", case_dir)
        if first_path is None:
            self.write(400, page("<section><p>First image is required.</p></section>"))
            return

        result = analyze_case(self.artifact, self.model, self.transform, case_id, operator_id, first_path, second_path, quality_ok)
        persist_case(case_id, result)
        decision = result["decision"]
        self.write(
            200,
            page(
                f"""
<section>
  <h2>Case Complete</h2>
  <p class="ok">Report generated for <code>{html.escape(case_id)}</code>.</p>
  <p><strong>Action:</strong> {html.escape(decision['action'])}</p>
  <p><strong>Recommendation:</strong> {html.escape(decision['recommendation'])}</p>
  <p><strong>Referable DR score:</strong> {decision['referable_score']:.4f}</p>
  <p><strong>Manual review:</strong> required</p>
  <p><a href="/case?id={html.escape(case_id)}">Open clinician report</a></p>
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
            )
            case_dir = persist_case(case_id, result)
            result = {
                **result,
                "artifacts": {
                    "case_dir": str(case_dir),
                    "result_json": str(case_dir / "result.json"),
                    "report_html": str(case_dir / "report.html"),
                    "report_url": f"/case?id={case_id}",
                },
            }
            self.write(200, json.dumps(result, indent=2, sort_keys=True).encode("utf-8"), "application/json; charset=utf-8")
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
    assert b"ActiveRetina-RC Clinical Station" in form_page()
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
