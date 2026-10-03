"""Evaluate a timm-style Hugging Face checkpoint under the 90/0 gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .formal_risk_control import binomial_upper


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, project_root: Path) -> None:
        self.rows = rows
        self.transform = transform
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.rows)

    def resolve(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.exists():
            return p
        c = self.project_root / raw_path
        if c.exists():
            return c
        normalized = raw_path.replace("\\", "/")
        marker = "/external/"
        if marker in normalized:
            c = self.project_root / "external" / normalized.split(marker, 1)[1]
            if c.exists():
                return c
        return p

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.resolve(row["image_path"])) as img:
            image = img.convert("RGB")
        return self.transform(image), row["image_id"], int(row["grade"]), int(row["referable_dr"])


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    state = {}
    for key, value in payload.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    return state


def download_checkpoint(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id, filename, local_dir=local_dir))


def build_model(model_name: str, checkpoint: Path, image_size: int, num_classes: int):
    import timm

    model = timm.create_model(model_name, pretrained=False, img_size=image_size, num_classes=num_classes)
    state = load_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    cfg = timm.data.resolve_model_data_config(model)
    cfg["input_size"] = (3, image_size, image_size)
    cfg["mean"] = (0.485, 0.456, 0.406)
    cfg["std"] = (0.229, 0.224, 0.225)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, {"missing": list(missing), "unexpected": list(unexpected)}


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, positive_indices: list[int], use_amp: bool):
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, image_ids, grades, ys in loader:
            with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
                logits = model(images.to(device))
                probs = torch.softmax(logits, dim=1)
                scores = probs[:, positive_indices].sum(dim=1)
            probs_np = probs.detach().cpu().float().numpy()
            scores_np = scores.detach().cpu().float().numpy()
            for image_id, grade, y, score, prob in zip(image_ids, grades.numpy(), ys.numpy(), scores_np, probs_np):
                rows.append(
                    {
                        "image_id": str(image_id),
                        "grade": int(grade),
                        "referable_dr": int(y),
                        "score": float(score),
                        "probs": [float(x) for x in prob],
                    }
                )
    return rows


def eval_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    pred = scores >= threshold
    decided_pos = int(((y == 1) & pred).sum())
    miss = int(((y == 1) & (~pred)).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & (~pred)).sum())
    ucb = binomial_upper(miss, decided_pos, 0.05) if decided_pos else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(rows)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "decided_positive": decided_pos,
        "missed_positive": miss,
        "false_positive": fp,
        "true_negative": tn,
        "specificity": float(tn / max(1, int((y == 0).sum()))),
        "referable_action_rate": float(pred.mean()),
        "missed_risk_ucb95": ucb,
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def select_threshold(rows: list[dict[str, Any]], factors: list[float], include_observed: bool, min_positive: int):
    y = np.asarray([r["referable_dr"] for r in rows], dtype=int)
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    candidates: list[tuple[str, float]] = []
    if include_observed:
        candidates.extend((f"observed_{i}", float(v)) for i, v in enumerate(sorted(set(scores))))
    pos = scores[y == 1]
    if len(pos):
        floor = float(pos.min())
        candidates.extend((f"positive_floor_x{f:g}", max(0.0, floor * f)) for f in factors)
    best = None
    for name, thr in candidates:
        m = eval_threshold(rows, thr)
        m["threshold_name"] = name
        if m["missed_positive"] != 0 or m["decided_positive"] < min_positive:
            continue
        if best is None or (m["specificity"], m["decided_positive"], -m["referable_action_rate"]) > (
            best["specificity"],
            best["decided_positive"],
            -best["referable_action_rate"],
        ):
            best = m
    return best or {"threshold": 1.01, "threshold_name": "no_valid_zero_miss_threshold"}


def write_predictions(path: Path, split: str, rows: list[dict[str, Any]], threshold: float) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "grade", "referable_dr", "score", "action", "probs_json"])
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split,
                    "image_id": row["image_id"],
                    "grade": row["grade"],
                    "referable_dr": row["referable_dr"],
                    "score": f"{row['score']:.8f}",
                    "action": "AUTONOMOUS_REFERABLE" if row["score"] >= threshold else "AUTONOMOUS_NONREFERABLE_CANDIDATE",
                    "probs_json": json.dumps(row["probs"], separators=(",", ":")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-filename", default="checkpoint-best.pth")
    parser.add_argument("--model-name", default="vit_large_patch16_224")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--positive-indices", default="2,3,4")
    parser.add_argument("--min-specificity", type=float, default=0.90)
    parser.add_argument("--min-validation-positive", type=int, default=59)
    parser.add_argument("--threshold-safety-factors", default="0.999,0.99,0.95,0.9,0.8,0.7,0.5,0.25")
    parser.add_argument("--include-observed-thresholds", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = download_checkpoint(args.repo_id, args.checkpoint_filename, Path("external/hf_checkpoints") / args.repo_id.replace("/", "__"))
    device = device_for(args.device)
    model, transform, load_report = build_model(args.model_name, ckpt, args.image_size, args.num_classes)
    model.to(device)
    rows = read_manifest(Path(args.manifest))
    by_split = split_rows(rows)
    project_root = Path.cwd()
    positive_indices = [int(x) for x in args.positive_indices.split(",") if x.strip()]
    valid_loader = DataLoader(FundusDataset(by_split["valid"], transform, project_root), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FundusDataset(by_split["test"], transform, project_root), batch_size=args.batch_size, shuffle=False)
    valid = predict(model, valid_loader, device, positive_indices, args.use_amp)
    selected = select_threshold(
        valid,
        [float(x) for x in args.threshold_safety_factors.split(",") if x.strip()],
        args.include_observed_thresholds,
        args.min_validation_positive,
    )
    threshold = float(selected["threshold"])
    valid_metrics = eval_threshold(valid, threshold)
    test = predict(model, test_loader, device, positive_indices, args.use_amp)
    test_metrics = eval_threshold(test, threshold)
    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid, threshold)
    write_predictions(pred_path, "test", test, threshold)
    summary = {
        "status": (
            "GO_TIMM_HF_CHECKPOINT_90_SPECIFICITY_ZERO_MISS"
            if test_metrics["missed_positive"] == 0 and test_metrics["specificity"] >= args.min_specificity
            else "NO_GO_TIMM_HF_CHECKPOINT_90_SPECIFICITY_ZERO_MISS"
        ),
        "repo_id": args.repo_id,
        "checkpoint": str(ckpt),
        "model_name": args.model_name,
        "image_size": args.image_size,
        "positive_indices": positive_indices,
        "load_report": load_report,
        "selected_threshold": selected,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "target_specificity": args.min_specificity,
        "predictions": str(pred_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
