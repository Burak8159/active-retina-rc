"""Local-only upload UI and JSON API for the RFMiD multi-label research model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


DEFAULT_CHECKPOINT = Path("models/rfmid_convnext_tiny_multilabel_20261021.pt")
DEFAULT_SUMMARY = Path("results/rfmid_multidisease_convnext_tiny_20261021/run_summary.json")
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
THRESHOLD = 0.5
DISCLAIMER = (
    "Research-only output. A model flag is not a diagnosis and must not be used "
    "to make patient-care decisions. This model has not been clinically validated."
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(checkpoint_path: Path, summary_path: Path, requested_device: str = "auto") -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "torchvision.convnext_tiny":
        raise ValueError("Unsupported model architecture in checkpoint.")
    labels = checkpoint.get("labels")
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise ValueError("Checkpoint label list is missing or invalid.")

    model = models.convnext_tiny(weights=None)
    final = model.classifier[2]
    model.classifier[2] = torch.nn.Linear(final.in_features, len(labels))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = choose_device(requested_device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model.to(device).eval()

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_label = summary.get("test_metrics", {}).get("per_label", {})
    image_size = int(checkpoint.get("input_size", 320))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return {
        "model": model,
        "transform": transform,
        "labels": labels,
        "per_label_reference": per_label,
        "device": str(device),
        "architecture": checkpoint["architecture"],
        "epoch": int(checkpoint.get("epoch", -1)),
        "input_size": image_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "lock": threading.Lock(),
    }


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Choose an image to analyze.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPEG, PNG, or WebP fundus image.")
            if source.width * source.height > 40_000_000:
                raise ValueError("Image dimensions are too large.")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The upload is not a readable fundus image file.") from exc


def make_response(bundle: dict[str, Any], scores: list[float], case_id: str | None = None) -> dict[str, Any]:
    if len(scores) != len(bundle["labels"]):
        raise ValueError("Model output count does not match checkpoint labels.")
    outputs = []
    for label, score in zip(bundle["labels"], scores):
        ref = bundle["per_label_reference"].get(label, {})
        outputs.append({
            "label": label,
            "score": round(float(score), 6),
            "threshold": THRESHOLD,
            "model_flag": bool(score >= THRESHOLD),
            "output": "FLAGGED" if score >= THRESHOLD else "NOT_FLAGGED",
            "rfmid_test_reference": {
                "n": ref.get("n"),
                "positive": ref.get("positive"),
                "accuracy_at_0_5": ref.get("accuracy_at_0_5"),
                "sensitivity_at_0_5": ref.get("sensitivity_at_0_5"),
                "specificity_at_0_5": ref.get("specificity_at_0_5"),
                "auroc": ref.get("auroc"),
                "average_precision": ref.get("average_precision"),
                "false_negatives": ref.get("confusion", {}).get("fn"),
            } if ref else None,
        })
    risk = next((item for item in outputs if item["label"] == "Disease_Risk"), None)
    conditions = [item for item in outputs if item["label"] != "Disease_Risk"]
    return {
        "case_id": case_id or str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "architecture": bundle["architecture"],
            "selected_epoch": bundle["epoch"],
            "checkpoint_sha256": bundle["checkpoint_sha256"],
            "device": bundle["device"],
            "threshold": THRESHOLD,
            "status": "RFMiD_DEVELOPMENT_MODEL_NOT_CLINICALLY_VALIDATED",
        },
        "dataset_overall_risk_label": risk,
        "conditions": conditions,
        "condition_outputs_are_independent": True,
        "image_saved": False,
        "autonomous_diagnosis": False,
        "disclaimer": DISCLAIMER,
    }


def analyze_bytes(bundle: dict[str, Any], image_bytes: bytes) -> dict[str, Any]:
    image = decode_image(image_bytes)
    tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])
    with bundle["lock"], torch.inference_mode():
        logits = bundle["model"](tensor)
        scores = torch.sigmoid(logits.float()).squeeze(0).cpu().tolist()
    return make_response(bundle, scores)


def app_page() -> bytes:
    return PAGE.encode("utf-8")


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>ActiveRetina | Research Scan</title>
<style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#18292d;background:#f3f6f5}
*{box-sizing:border-box}body{margin:0}header{background:#123d3b;color:#fff;padding:14px 22px;display:flex;align-items:center;justify-content:space-between}
.brand{font-size:19px;font-weight:700}.tag{font-size:12px;color:#d5e8e4;border:1px solid #6f9f98;padding:5px 8px;border-radius:3px}
main{max-width:1120px;margin:0 auto;padding:24px 20px 48px}.notice{background:#fff6e4;border-left:4px solid #cc8b19;padding:12px 15px;line-height:1.45;font-size:14px;margin-bottom:18px}
.scan{background:white;border:1px solid #d7e0dd;padding:18px;display:grid;grid-template-columns:minmax(230px,1fr) auto;gap:14px;align-items:end}
label{font-size:13px;font-weight:650;display:block;margin-bottom:7px}input[type=file]{width:100%;padding:10px;border:1px solid #b9c7c4;border-radius:3px;background:#fff}
button{background:#087a71;color:white;border:0;border-radius:3px;padding:12px 18px;font-size:14px;font-weight:700;cursor:pointer;min-height:42px}button:disabled{opacity:.6;cursor:wait}
.status{min-height:24px;margin:12px 0;color:#526563;font-size:14px}.overall{display:none;background:white;border-top:3px solid #237f70;padding:14px 16px;margin:18px 0}.overall h2{font-size:16px;margin:0 0 8px}.overall p{margin:0;font-size:14px}
.toolbar{display:flex;justify-content:space-between;align-items:center;margin:20px 0 8px;gap:12px}.toolbar h2{font-size:17px;margin:0}.toolbar input{max-width:280px;padding:9px 10px;border:1px solid #b9c7c4;border-radius:3px}
.tablewrap{overflow-x:auto;background:white;border:1px solid #d7e0dd}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e4eae8;white-space:nowrap}th{background:#eaf0ee;color:#354e4a;font-weight:650}td.score{font-variant-numeric:tabular-nums}.flag{font-weight:700;color:#9b3b26}.clear{font-weight:650;color:#28695c}.muted{color:#71817e}
.foot{font-size:12px;color:#5c6b68;line-height:1.5;margin-top:12px}.hidden{display:none!important}
@media(max-width:650px){header{padding:12px 15px}.tag{max-width:170px;text-align:center}main{padding:16px 12px 32px}.scan{grid-template-columns:1fr}.toolbar{align-items:flex-start;flex-direction:column}.toolbar input{max-width:none;width:100%}th,td{padding:9px 8px}}
</style></head><body>
<header><div class="brand">ActiveRetina</div><div class="tag">RFMiD research model</div></header>
<main>
<div class="notice"><strong>Research use only.</strong> Outputs are independent model flags, not diagnoses. This model is not clinically validated and must not guide care decisions. The 0.5 threshold is a development threshold, not a clinical cutoff.</div>
<form id="scanForm" class="scan"><div><label for="image">Fundus image</label><input id="image" name="image" type="file" accept="image/jpeg,image/png,image/webp" required></div><button id="submit" type="submit">Analyze image</button></form>
<p id="status" class="status" role="status" aria-live="polite"></p>
<section id="overall" class="overall"><h2>Dataset overall-risk label</h2><p id="riskText"></p></section>
<div class="toolbar"><h2>Per-condition model outputs</h2><input id="filter" type="search" placeholder="Filter condition codes" aria-label="Filter condition codes"></div>
<div class="tablewrap"><table><thead><tr><th>Condition code</th><th>Model score</th><th>Model output</th><th>RFMiD test positives</th><th>Test sensitivity</th><th>Test specificity</th><th>Test AUROC</th></tr></thead><tbody id="rows"><tr><td colspan="7" class="muted">Upload an image to see separate outputs for each condition.</td></tr></tbody></table></div>
<p class="foot">Scores are not calibrated probabilities. Reference metrics describe this model's RFMiD image-level test split only; patient-level separation was not verifiable. They do not establish performance on a new patient, site, camera, or clinical population. Images are processed in memory and are not saved by this station.</p>
</main>
<script>
const form=document.getElementById('scanForm'),statusEl=document.getElementById('status'),rows=document.getElementById('rows'),overall=document.getElementById('overall'),riskText=document.getElementById('riskText'),filter=document.getElementById('filter'),button=document.getElementById('submit');
let outputs=[];
function pct(v){return v==null?'—':(100*v).toFixed(1)+'%'}
function render(){const q=filter.value.trim().toLowerCase();const list=outputs.filter(x=>x.label.toLowerCase().includes(q));rows.replaceChildren();if(!list.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.className='muted';td.textContent='No matching condition codes.';tr.append(td);rows.append(tr);return}for(const x of list){const ref=x.rfmid_test_reference||{},tr=document.createElement('tr');const cells=[x.label,(100*x.score).toFixed(1)+'%',x.output,ref.positive==null?'—':`${ref.positive} / ${ref.n}`,pct(ref.sensitivity_at_0_5),pct(ref.specificity_at_0_5),ref.auroc==null?'—':Number(ref.auroc).toFixed(3)];cells.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===1)td.className='score';if(i===2)td.className=x.model_flag?'flag':'clear';tr.append(td)});rows.append(tr)}}
filter.addEventListener('input',render);
form.addEventListener('submit',async e=>{e.preventDefault();const file=document.getElementById('image').files[0];if(!file)return;if(file.size>16*1024*1024){statusEl.textContent='File is over the 16 MB upload limit.';return}button.disabled=true;statusEl.textContent='Analyzing…';overall.style.display='none';try{const data=new FormData();data.append('image',file);const response=await fetch('/api/predict',{method:'POST',body:data});const result=await response.json();if(!response.ok)throw new Error(result.error||'Analysis failed');outputs=result.conditions;render();const risk=result.dataset_overall_risk_label,ref=risk?.rfmid_test_reference;riskText.textContent=risk?`${risk.output} — model score ${(100*risk.score).toFixed(1)}%. RFMiD test reference: ${pct(ref?.accuracy_at_0_5)} accuracy, ${pct(ref?.sensitivity_at_0_5)} sensitivity, ${pct(ref?.specificity_at_0_5)} specificity; ${ref?.false_negatives??'—'} of ${ref?.positive??'—'} positive images missed. Dataset-specific image-level result, not a universal retinal-disease assessment.`:'No overall-risk output available.';overall.style.display='block';statusEl.textContent='Analysis complete. No image was saved.'}catch(err){statusEl.textContent=err.message||'Could not analyze this image.'}finally{button.disabled=false}});
</script></body></html>'''


class StationHandler(BaseHTTPRequestHandler):
    bundle: dict[str, Any]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code: int, value: dict[str, Any]) -> None:
        self.send_bytes(code, json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_bytes(200, app_page(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self.send_json(200, {"status": "ready", "model": self.bundle["architecture"],
                                 "device": self.bundle["device"], "autonomous_diagnosis": False})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid_content_length"})
            return
        if length <= 0:
            self.send_json(400, {"error": "empty_request"})
            return
        if length > MAX_UPLOAD_BYTES + 128 * 1024:
            self.send_json(413, {"error": "upload_too_large", "limit_bytes": MAX_UPLOAD_BYTES})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self.send_json(415, {"error": "multipart_form_data_required"})
            return
        body = self.rfile.read(length)
        try:
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
            )
            image_bytes = None
            for part in message.iter_parts():
                if part.get_content_disposition() == "form-data" and part.get_param("name", header="content-disposition") == "image":
                    image_bytes = part.get_payload(decode=True)
                    break
            if image_bytes is None or len(image_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError("Upload one image no larger than 16 MB.")
            result = analyze_bytes(self.bundle, image_bytes)
            self.send_json(200, result)
        except (ValueError, UnidentifiedImageError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "inference_failed"})


def run(host: str, port: int, checkpoint: Path, summary: Path, device: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This research station binds to loopback only; remote access is disabled.")
    StationHandler.bundle = load_bundle(checkpoint, summary, device)
    server = ThreadingHTTPServer((host, port), StationHandler)
    print(f"RFMiD research station: http://{host}:{port}", flush=True)
    print(f"Model: {StationHandler.bundle['architecture']} on {StationHandler.bundle['device']}; "
          f"checkpoint SHA-256 {StationHandler.bundle['checkpoint_sha256']}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    run(args.host, args.port, args.checkpoint, args.summary, args.device)


if __name__ == "__main__":
    main()
