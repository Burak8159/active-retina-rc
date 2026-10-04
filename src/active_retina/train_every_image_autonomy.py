"""Train an every-image autonomous DR classifier.

This script targets the hard product requirement directly:

* every image receives a binary CLEAR/REFER decision,
* validation threshold selection requires zero referable cases cleared,
* the frozen threshold is applied once to test,
* GO requires test false clearances = 0 and test specificity >= target.

It differs from the earlier refer/defer and selective-clearance scripts by using
an ordinal DR-grade head. The referable score is the predicted probability of
grade >= 2, so the model is trained around the clinical boundary rather than
only with a binary BCE surrogate.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(20261004 + len(rows) + limit)
    positives = [r for r in rows if int(r["referable_dr"]) == 1]
    negatives = [r for r in rows if int(r["referable_dr"]) == 0]
    n_pos = min(len(positives), max(1, limit // 2))
    n_neg = min(len(negatives), limit - n_pos)
    chosen = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(chosen)
    return chosen


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any, root: Path) -> None:
        self.rows = rows
        self.transform = transform
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = self.resolve_image_path(row["image_path"])
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return (
            self.transform(image),
            torch.tensor(int(row["grade"]), dtype=torch.long),
            torch.tensor(int(row["referable_dr"]), dtype=torch.long),
            row["image_id"],
        )

    def resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        candidate = self.root / raw_path
        if candidate.exists():
            return candidate
        normalized = raw_path.replace("\\", "/")
        marker = "/external/"
        if marker in normalized:
            candidate = self.root / "external" / normalized.split(marker, 1)[1]
            if candidate.exists():
                return candidate
        return path


class GradeReferableModel(nn.Module):
    def __init__(self, model_name: str, image_size: int | None, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=6)
        cfg = timm.data.resolve_model_data_config(self.model)
        if image_size:
            cfg["input_size"] = (3, image_size, image_size)
        self.train_transform = timm.data.create_transform(**cfg, is_training=True)
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {split: [r for r in rows if r["source_split"] == f"ddr_{split}"] for split in ("train", "valid", "test")}


def set_trainable(model: nn.Module, fragments: list[str]) -> dict[str, Any]:
    if not fragments:
        return {
            "mode": "all_parameters",
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        keep = any(fragment in name for fragment in fragments)
        param.requires_grad = keep
        if keep:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return {"mode": "filtered", "fragments": fragments, "trainable_parameters": trainable, "frozen_parameters": frozen}


def class_weights(rows: list[dict[str, Any]], grade2_boost: float, grade1_boost: float) -> torch.Tensor:
    counts = np.ones(6, dtype=float)
    for row in rows:
        counts[int(row["grade"])] += 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    weights[2] *= grade2_boost
    weights[1] *= grade1_boost
    return torch.tensor(weights, dtype=torch.float32)


def sampler_weights(rows: list[dict[str, Any]], grade2_boost: float, grade1_boost: float) -> list[float]:
    out = []
    for row in rows:
        grade = int(row["grade"])
        referable = int(row["referable_dr"])
        weight = 1.0
        if referable:
            weight *= 3.0
        if grade == 2:
            weight *= grade2_boost
        if grade == 1:
            weight *= grade1_boost
        out.append(weight)
    return out


def referable_score(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return probs[:, 2:].sum(dim=1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    grades: list[int] = []
    labels: list[int] = []
    scores: list[float] = []
    ids: list[str] = []
    with torch.inference_mode():
        for images, grade, referable, image_ids in loader:
            logits = model(images.to(device))
            score = referable_score(logits).detach().cpu().float().numpy()
            scores.extend(float(x) for x in score)
            grades.extend(int(x) for x in grade.numpy())
            labels.extend(int(x) for x in referable.numpy())
            ids.extend(str(x) for x in image_ids)
    return np.asarray(labels, dtype=int), np.asarray(grades, dtype=int), np.asarray(scores, dtype=float), ids


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    refer = scores >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pos & refer).sum())
    fn = int((pos & ~refer).sum())
    tn = int((neg & ~refer).sum())
    fp = int((neg & refer).sum())
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(pos.sum()),
        "negative": int(neg.sum()),
        "true_positive": tp,
        "false_negative_clearance": fn,
        "true_negative": tn,
        "false_positive_referral": fp,
        "sensitivity": float(tp / max(1, int(pos.sum()))),
        "specificity": float(tn / max(1, int(neg.sum()))),
        "refer_rate": float(refer.mean()),
    }


def select_zero_miss_threshold(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = scores[y == 1]
    if len(positives) == 0:
        threshold = float(scores.max()) + 1e-8
    else:
        threshold = float(positives.min())
    return metrics(y, scores, threshold)


def write_predictions(path: Path, split: str, ids: list[str], grades: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_id", "grade", "referable_dr", "referable_score", "threshold", "action"],
        )
        if not exists:
            writer.writeheader()
        for image_id, grade, yy, score in zip(ids, grades, y, scores):
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "grade": int(grade),
                    "referable_dr": int(yy),
                    "referable_score": f"{float(score):.8f}",
                    "threshold": f"{float(threshold):.8f}",
                    "action": "REFER" if score >= threshold else "CLEAR",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-name", default="tf_efficientnet_b0.ns_jft_in1k")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grade2-boost", type=float, default=12.0)
    parser.add_argument("--grade1-boost", type=float, default=4.0)
    parser.add_argument("--boundary-margin-weight", type=float, default=0.5)
    parser.add_argument("--trainable-name-fragments", default="")
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest))
    by_split = split_rows(rows)
    by_split["train"] = limit_rows(by_split["train"], args.max_train)
    by_split["valid"] = limit_rows(by_split["valid"], args.max_valid)
    by_split["test"] = limit_rows(by_split["test"], args.max_test)
    device = device_for(args.device)
    wrapper = GradeReferableModel(args.model_name, args.image_size or None, not args.no_pretrained)
    model = wrapper.model
    trainable_info = set_trainable(model, [x.strip() for x in args.trainable_name_fragments.split(",") if x.strip()])
    model.to(device)

    train_dataset = FundusDataset(by_split["train"], wrapper.train_transform, root)
    valid_dataset = FundusDataset(by_split["valid"], wrapper.eval_transform, root)
    test_dataset = FundusDataset(by_split["test"], wrapper.eval_transform, root)
    sampler = WeightedRandomSampler(
        sampler_weights(by_split["train"], args.grade2_boost, args.grade1_boost),
        num_samples=len(by_split["train"]),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ce = nn.CrossEntropyLoss(weight=class_weights(by_split["train"], args.grade2_boost, args.grade1_boost).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.use_amp and device.type == "cuda"))

    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for images, grade, referable, _ in train_loader:
            images = images.to(device)
            grade = grade.to(device)
            referable = referable.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.use_amp and device.type == "cuda")):
                logits = model(images)
                loss = ce(logits, grade)
                score = referable_score(logits)
                # Extra pressure on the binary boundary: grade-2 positives must
                # outrank grade-1 negatives.
                binary_boundary = torch.nn.functional.binary_cross_entropy(score, referable)
                loss = loss + args.boundary_margin_weight * binary_boundary
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(grade)
            seen += len(grade)
        y_valid, g_valid, s_valid, _ = predict(model, valid_loader, device)
        selected = select_zero_miss_threshold(y_valid, s_valid)
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "valid": selected}
        history.append(row)
        key = (
            int(selected["false_negative_clearance"] == 0),
            float(selected["specificity"]),
            float(selected["sensitivity"]),
        )
        old_key = (
            int(best_epoch["valid"]["false_negative_clearance"] == 0),
            float(best_epoch["valid"]["specificity"]),
            float(best_epoch["valid"]["sensitivity"]),
        ) if best_epoch else (-1, -1.0, -1.0)
        if key > old_key:
            best_epoch = copy.deepcopy(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    y_valid, g_valid, s_valid, valid_ids = predict(model, valid_loader, device)
    selected = select_zero_miss_threshold(y_valid, s_valid)
    threshold = float(selected["threshold"])
    valid_metrics = metrics(y_valid, s_valid, threshold)
    y_test, g_test, s_test, test_ids = predict(model, test_loader, device)
    test_metrics = metrics(y_test, s_test, threshold)

    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid_ids, g_valid, y_valid, s_valid, threshold)
    write_predictions(pred_path, "test", test_ids, g_test, y_test, s_test, threshold)

    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "pretrained": not args.no_pretrained,
            "state_dict": model.cpu().state_dict(),
            "threshold": threshold,
            "trainable_info": trainable_info,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        args.artifact,
    )
    summary = {
        "status": (
            "GO_EVERY_IMAGE_AUTONOMY_90_0"
            if test_metrics["false_negative_clearance"] == 0 and test_metrics["specificity"] >= args.target_specificity
            else "NO_GO_EVERY_IMAGE_AUTONOMY_90_0"
        ),
        "manifest": args.manifest,
        "artifact": args.artifact,
        "predictions": str(pred_path),
        "model_name": args.model_name,
        "pretrained": not args.no_pretrained,
        "image_size": args.image_size,
        "trainable_info": trainable_info,
        "n_train": len(by_split["train"]),
        "n_valid": len(by_split["valid"]),
        "n_test": len(by_split["test"]),
        "target_specificity": args.target_specificity,
        "selected_threshold": selected,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
