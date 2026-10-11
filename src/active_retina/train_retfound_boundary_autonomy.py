"""Adapt RETFound's last ViT block for eye-level DR grading and safe referral.

Only the patient-disjoint train and validation partitions are read. The test
partition is never opened. This is development training, not certification.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audit_retfound_embedding_autonomy import CHECKPOINT_GLOB, MODEL_NAME, grouped_eyes, read_development_rows, resolve_image


class EyeDataset(Dataset):
    def __init__(self, eyes: list[dict[str, Any]], transform: Any, root: Path) -> None:
        self.eyes, self.transform, self.root = eyes, transform, root

    def __len__(self) -> int:
        return len(self.eyes)

    def __getitem__(self, index: int):
        eye = self.eyes[index]
        tensors = []
        for row in eye["views"]:
            with Image.open(resolve_image(self.root, row["image_path"])) as image:
                tensors.append(self.transform(image.convert("L").convert("RGB")))
        return torch.stack(tensors), int(eye["grade"]), str(eye["source_dataset"]), str(eye["patient_id"])


def collate_eyes(batch):
    batch_size = len(batch)
    height, width = batch[0][0].shape[-2:]
    images = torch.zeros(batch_size, 2, 3, height, width, dtype=batch[0][0].dtype)
    mask = torch.zeros(batch_size, 2, dtype=torch.bool)
    grades = torch.empty(batch_size, dtype=torch.long)
    sources, patients = [], []
    for index, (views, grade, source, patient) in enumerate(batch):
        if len(views) not in (1, 2):
            raise ValueError(f"Expected one or two images per eye, got {len(views)}")
        images[index, :len(views)] = views
        mask[index, :len(views)] = True
        grades[index] = grade
        sources.append(source)
        patients.append(patient)
    return images, mask, grades, sources, patients


class AdaptedRetFound(nn.Module):
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        super().__init__()
        import timm
        from safetensors.torch import load_file

        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5, img_size=256, global_pool="avg")
        self.backbone.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
        old_head = self.backbone.head
        new_head = nn.Linear(old_head.in_features, 6)
        with torch.no_grad():
            new_head.weight[:5].copy_(old_head.weight)
            new_head.bias[:5].copy_(old_head.bias)
            nn.init.normal_(new_head.weight[5:], mean=0.0, std=0.002)
            new_head.bias[5].fill_(-2.0)
        self.backbone.head = new_head
        self.backbone.num_classes = 6
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone.blocks[-1].parameters():
            parameter.requires_grad_(True)
        for module_name in ("fc_norm", "head"):
            module = getattr(self.backbone, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        cfg = timm.data.resolve_model_data_config(self.backbone)
        cfg["input_size"] = (3, 256, 256)
        self.train_transform = timm.data.create_transform(
            **cfg, is_training=True, hflip=0.5, color_jitter=0.08, re_prob=0.0
        )
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)
        self.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep all frozen blocks deterministic; the last block and head train.
        for block in self.backbone.blocks[:-1]:
            block.eval()
        if mode:
            self.backbone.patch_embed.eval()
            self.backbone.pos_drop.eval()
            self.backbone.norm_pre.eval()
        return self

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = images.shape
        valid = images[mask]
        logits_valid = self.backbone(valid)
        logits = torch.zeros(batch, views, 6, device=logits_valid.device, dtype=logits_valid.dtype)
        logits[mask] = logits_valid
        weights = mask.unsqueeze(-1).to(logits.dtype)
        return (logits * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def outputs(logits: torch.Tensor) -> torch.Tensor:
    # Classes 0/1 clear only when low risk; 2-4 and explicit ungradable class 5 refer.
    return torch.softmax(logits.float(), dim=1)[:, 2:].sum(dim=1)


def validation_metrics(eyes: list[dict[str, Any]], score: np.ndarray, threshold: float) -> dict[str, Any]:
    grades = np.asarray([eye["grade"] for eye in eyes])
    refer = (grades >= 2) & (grades <= 4)
    ungradable = grades == 5
    negative = grades <= 1
    action = score >= threshold
    missed = refer & ~action
    missed_quality = ungradable & ~action
    return {
        "n_eyes": len(eyes),
        "n_gradable_nonreferable": int(negative.sum()),
        "n_referable": int(refer.sum()),
        "n_ungradable": int(ungradable.sum()),
        "true_negative_clearances": int((negative & ~action).sum()),
        "specificity": float((~action[negative]).mean()) if negative.any() else None,
        "missed_referable": int(missed.sum()),
        "missed_ungradable": int(missed_quality.sum()),
        "ungradable_refer_rate": float(action[ungradable].mean()) if ungradable.any() else None,
        "binary_action_coverage": 1.0,
        "referral_rate": float(action.mean()),
        "threshold": float(threshold),
        "by_source": {},
        "by_grade": {},
    } | stratified_metrics(eyes, grades, action, missed, missed_quality)


def stratified_metrics(eyes, grades, action, missed, missed_quality):
    result = {"by_source": {}, "by_grade": {}}
    sources = np.asarray([eye["source_dataset"] for eye in eyes])
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        neg, pos, bad = mask & (grades <= 1), mask & (grades >= 2) & (grades <= 4), mask & (grades == 5)
        result["by_source"][source] = {
            "negative_n": int(neg.sum()),
            "specificity": float((~action[neg]).mean()) if neg.any() else None,
            "referable_n": int(pos.sum()),
            "missed_referable": int(missed[pos].sum()),
            "ungradable_n": int(bad.sum()),
            "missed_ungradable": int(missed_quality[bad].sum()),
        }
    for grade in sorted(set(grades.tolist())):
        mask = grades == grade
        result["by_grade"][str(grade)] = {
            "n": int(mask.sum()),
            "referral_rate": float(action[mask].mean()),
        }
    return result


def predict(model, loader, device) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for images, mask, _grades, _sources, _patients in loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            values.extend(outputs(logits).cpu().numpy().tolist())
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--minimum-gate-specificity", type=float, default=0.3730569948)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = Path.cwd()
    matches = sorted(glob.glob(str(root / CHECKPOINT_GLOB)))
    checkpoint = Path(args.checkpoint) if args.checkpoint else (Path(matches[-1]) if matches else None)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"RETFound safetensors checkpoint not found: {CHECKPOINT_GLOB}")
    rows = read_development_rows(Path(args.manifest))
    train_eyes, valid_eyes = grouped_eyes(rows["train"]), grouped_eyes(rows["valid"])
    train_patients = {eye["patient_id"] for eye in train_eyes}
    valid_patients = {eye["patient_id"] for eye in valid_eyes}
    overlap = train_patients & valid_patients
    if overlap:
        raise ValueError(f"Patient leakage: {len(overlap)} patients overlap train/validation")
    model = AdaptedRetFound(checkpoint, device)
    train_data = EyeDataset(train_eyes, model.train_transform, root)
    valid_data = EyeDataset(valid_eyes, model.eval_transform, root)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", collate_fn=collate_eyes)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", collate_fn=collate_eyes)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, mask, grades, _sources, _patients in train_loader:
            images, mask, grades = images.to(device, non_blocking=True), mask.to(device, non_blocking=True), grades.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, mask).float()
                ce = nn.functional.cross_entropy(logits, grades)
                refer_target = ((grades >= 2) & (grades <= 4)) | (grades == 5)
                risk_logit = torch.logsumexp(logits[:, 2:], dim=1) - torch.logsumexp(logits[:, :2], dim=1)
                risk = nn.functional.binary_cross_entropy_with_logits(risk_logit, refer_target.float())
                gradable = grades < 5
                cumulative = []
                for boundary in range(1, 5):
                    boundary_logit = torch.logsumexp(logits[:, boundary:5], dim=1) - torch.logsumexp(logits[:, :boundary], dim=1)
                    target = (grades >= boundary).float()
                    cumulative.append(nn.functional.binary_cross_entropy_with_logits(boundary_logit[gradable], target[gradable]))
                ordinal = torch.stack(cumulative).mean() if bool(gradable.any()) else logits.sum() * 0.0
                loss = ce + risk + 0.5 * ordinal
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(grades)
            seen += len(grades)
        scores = predict(model, valid_loader, device)
        grades = np.asarray([eye["grade"] for eye in valid_eyes])
        required = ((grades >= 2) & (grades <= 4)) | (grades == 5)
        if not required.any():
            raise ValueError("Validation requires positive referable/ungradable cases")
        threshold = float(scores[required].min() - 1e-12)
        metrics = validation_metrics(valid_eyes, scores, threshold)
        record = {"epoch": epoch, "train_loss": total_loss / max(1, seen), "validation": metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or (metrics["specificity"], -metrics["referral_rate"]) > (best["validation"]["specificity"], -best["validation"]["referral_rate"]):
            best = record
            torch.save({"state_dict": model.state_dict(), "best_epoch": best, "args": vars(args),
                        "claim_boundary": "DEVELOPMENT_ONLY_NOT_CERTIFIED"}, args.artifact)
        (out / "progress.json").write_text(json.dumps({"status": "TRAINING_IN_PROGRESS", "history": history}, indent=2) + "\n")
        if epoch == 1 and metrics["specificity"] < args.minimum_gate_specificity:
            gate = {"status": "STOP_EPOCH1_GATE_NOT_MET", "baseline_specificity": args.minimum_gate_specificity,
                    "observed_specificity": metrics["specificity"], "history": history}
            (out / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
            break
    final_status = "COMPLETE" if len(history) == args.epochs else "STOPPED_BY_EPOCH1_GATE"
    summary = {
        "status": final_status,
        "claim_boundary": "All available datasets have informed development. Only patient-disjoint train/validation were loaded; untouched certification and clinical validation are absent.",
        "backbone": "RETFound ViT-L/16 EyePACS five-grade checkpoint; all weights strict-loaded; final transformer block, fc_norm and expanded six-class head trainable; earlier blocks frozen.",
        "expanded_class_5": "ungradable; trained only from DDR grade-5 labels; predictions are model outputs, not ground-truth overrides",
        "loss": "six-class cross entropy + referable-or-ungradable binary BCE + 0.5 cumulative ordinal BCE across grade thresholds 1-4",
        "train_eyes": len(train_eyes), "validation_eyes": len(valid_eyes),
        "patient_overlap": len(overlap), "trainable_parameters": sum(p.numel() for p in params),
        "test_images_loaded": 0, "history": history, "best_epoch": best,
        "artifact": args.artifact,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "progress.json").write_text(json.dumps({"status": final_status, "epochs_completed": len(history)}, indent=2) + "\n")
    print(json.dumps({"status": final_status, "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
