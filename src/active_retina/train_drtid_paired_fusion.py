"""Patient-disjoint development experiment for paired-view DRTiD fusion.

Uses only the official DRTiD training partition, split by patient into
development-train and validation. The official testing partition is excluded:
it has already been inspected elsewhere and cannot certify this experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


FIELDS = ("macula", "optic_disc")


def read_pairs(manifest: Path, root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source = str(row.get("source_dataset", "")).lower()
            split = str(row.get("source_split", "")).lower()
            if source and source not in {"drtid", "drtid_paired"}:
                continue
            if not (split.endswith("training") or split.endswith("_train") or split in {"train", "training"}):
                continue
            eye = grouped.setdefault(
                row["eye_id"],
                {
                    "patient_id": row["patient_id"],
                    "eye_id": row["eye_id"],
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "views": {},
                },
            )
            if eye["patient_id"] != row["patient_id"] or eye["grade"] != int(row["grade"]):
                raise ValueError(f"Conflicting metadata for eye {row['eye_id']}")
            eye["views"][row["field"]] = row["image_path"]

    pairs = [item for item in grouped.values() if all(field in item["views"] for field in FIELDS)]
    if not pairs:
        raise ValueError("No complete macula/optic-disc pairs in the training partition.")
    for item in pairs:
        for field in FIELDS:
            image = resolve_path(root, item["views"][field])
            if not image.is_file():
                raise FileNotFoundError(f"Missing {field} image for {item['eye_id']}: {image}")
            item["views"][field] = str(image)
    return pairs


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    return root / path


def patient_split(pairs: list[dict[str, Any]], validation_fraction: float, seed: int):
    patient_labels: dict[str, int] = {}
    for item in pairs:
        patient = str(item["patient_id"])
        patient_labels[patient] = max(patient_labels.get(patient, 0), int(item["referable_dr"]))
    rng = np.random.default_rng(seed)
    validation_patients: set[str] = set()
    for label in (0, 1):
        patients = sorted(p for p, y in patient_labels.items() if y == label)
        rng.shuffle(patients)
        n_valid = max(1, int(round(len(patients) * validation_fraction))) if len(patients) > 1 else 0
        validation_patients.update(patients[:n_valid])
    train = [item for item in pairs if str(item["patient_id"]) not in validation_patients]
    valid = [item for item in pairs if str(item["patient_id"]) in validation_patients]
    if not train or not valid:
        raise ValueError("Patient split produced an empty train or validation set.")
    return train, valid


class PairedFundusDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]], transform: Any) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        item = self.pairs[index]
        tensors = []
        for field in FIELDS:
            with Image.open(item["views"][field]) as image:
                tensors.append(self.transform(image.convert("RGB")))
        return (
            torch.stack(tensors),
            int(item["grade"]),
            int(item["referable_dr"]),
            index,
        )


class DeepDRiDQualityDataset(Dataset):
    """Paired DeepDRiD images with the dataset's per-view overall-quality labels."""

    def __init__(self, manifest: Path, root: Path, transform: Any, allowed_splits: set[str]) -> None:
        self.items = []
        with manifest.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("source_split") not in allowed_splits:
                    continue
                views = row.get("views", {})
                if not all(field in views for field in FIELDS):
                    continue
                paths = {field: resolve_path(root, views[field]["image_path"]) for field in FIELDS}
                if not all(path.is_file() for path in paths.values()):
                    raise FileNotFoundError(f"Missing DeepDRiD quality image for {row.get('eye_id')}")
                quality_labels = [int(views[field]["quality"]["overall"]) for field in FIELDS]
                if any(value not in (0, 1) for value in quality_labels):
                    raise ValueError(f"Unexpected DeepDRiD overall-quality label for {row.get('eye_id')}: {quality_labels}")
                self.items.append({
                    "patient_id": str(row["patient_id"]),
                    "eye_id": str(row["eye_id"]),
                    "referable_dr": int(row["label"]["referable_dr"]),
                    "views": {field: str(paths[field]) for field in FIELDS},
                    "quality": quality_labels,
                })
        self.transform = transform
        if not self.items:
            raise ValueError(f"No paired DeepDRiD quality labels found for splits {sorted(allowed_splits)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        tensors = []
        for field in FIELDS:
            with Image.open(item["views"][field]) as image:
                tensors.append(self.transform(image.convert("RGB")))
        return torch.stack(tensors), torch.tensor(item["quality"], dtype=torch.float32), index


class PairedFusionModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, unfreeze_layer4: bool) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        width = int(self.encoder.num_features)
        if unfreeze_layer4:
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad = name.startswith("layer4.")
        self.view_heads = nn.ModuleList(nn.Linear(width, 5) for _ in FIELDS)
        self.quality_heads = nn.ModuleList(nn.Linear(width, 1) for _ in FIELDS)
        self.fusion = nn.Sequential(
            nn.Linear(width * 4, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 5),
        )

    def forward(self, images: torch.Tensor):
        batch, views, channels, height, width = images.shape
        embeddings = self.encoder(images.reshape(batch * views, channels, height, width))
        embeddings = embeddings.reshape(batch, views, -1)
        view_logits = torch.stack([head(embeddings[:, i]) for i, head in enumerate(self.view_heads)], dim=1)
        quality_logits = torch.stack(
            [head(embeddings[:, i]).squeeze(-1) for i, head in enumerate(self.quality_heads)], dim=1
        )
        first, second = embeddings[:, 0], embeddings[:, 1]
        fused_logits = self.fusion(torch.cat((first, second, torch.abs(first - second), first * second), dim=1))
        return fused_logits, view_logits, quality_logits


def referable_score(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits[:, 2:], dim=1) - torch.logsumexp(logits[:, :2], dim=1)


def ordinal_loss(logits: torch.Tensor, grade: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    cumulative = torch.stack([probabilities[:, threshold + 1 :].sum(dim=1) for threshold in range(4)], dim=1)
    targets = torch.stack([grade > threshold for threshold in range(4)], dim=1).float()
    return nn.functional.binary_cross_entropy(cumulative.clamp(1e-6, 1 - 1e-6), targets)


def hard_boundary_rank_loss(
    logits: torch.Tensor, grade: torch.Tensor, margin: float, top_fraction: float, grade2_weight: float
) -> torch.Tensor:
    scores = referable_score(logits.float())
    positives = grade >= 2
    negatives = grade < 2
    if not positives.any() or not negatives.any():
        return scores.new_tensor(0.0)
    pair_loss = nn.functional.softplus(margin - (scores[positives, None] - scores[None, negatives]))
    weights = torch.where(
        grade[positives, None] == 2,
        torch.full_like(pair_loss, grade2_weight),
        torch.ones_like(pair_loss),
    )
    weighted = pair_loss * weights
    k = max(1, int(np.ceil(weighted.numel() * top_fraction)))
    return torch.topk(weighted.flatten(), k=k, largest=True).values.mean()


def safe_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if not len(positives) or not len(negatives):
        return {"n": int(len(y)), "positive": int(len(positives)), "negative": int(len(negatives))}
    threshold = float(positives.min())
    specificity = float(np.mean(negatives < threshold))
    result = {
        "n": int(len(y)),
        "positive": int(len(positives)),
        "negative": int(len(negatives)),
        "threshold_zero_miss": threshold,
        "misses_at_threshold": int(np.sum(positives < threshold)),
        "specificity_at_zero_miss": specificity,
        "coverage": 1.0,
        "positive_score_min": float(positives.min()),
        "negative_score_median": float(np.median(negatives)),
    }
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    group_end = np.cumsum(counts)
    average_ranks = group_end - (counts - 1) / 2.0
    positive_rank_sum = float(average_ranks[inverse][y == 1].sum())
    result["auroc"] = (positive_rank_sum - len(positives) * (len(positives) + 1) / 2) / (
        len(positives) * len(negatives)
    )
    descending = np.argsort(-scores, kind="mergesort")
    sorted_positive = y[descending] == 1
    precision_at_rank = np.cumsum(sorted_positive) / np.arange(1, len(y) + 1)
    result["average_precision"] = float(precision_at_rank[sorted_positive].mean())
    return result


def aggregate_patient(pairs: list[dict[str, Any]], scores: np.ndarray):
    grouped: dict[str, dict[str, Any]] = {}
    for item, score in zip(pairs, scores):
        patient = str(item["patient_id"])
        bucket = grouped.setdefault(patient, {"label": 0, "score": -np.inf})
        bucket["label"] = max(bucket["label"], int(item["referable_dr"]))
        bucket["score"] = max(bucket["score"], float(score))
    labels = np.asarray([v["label"] for v in grouped.values()], dtype=np.int64)
    values = np.asarray([v["score"] for v in grouped.values()], dtype=np.float64)
    return safe_metrics(labels, values)


def score_model(model, loader, device, n: int):
    fused_scores = np.empty(n, dtype=np.float64)
    per_view_scores = np.empty((n, 2), dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    grades = np.empty(n, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for images, grade, label, indices in loader:
            images = images.to(device)
            fused, per_view, _quality = model(images)
            scores = referable_score(fused).float().cpu().numpy()
            branch_scores = torch.stack([referable_score(per_view[:, i]) for i in range(2)], dim=1).float().cpu().numpy()
            index = indices.numpy()
            fused_scores[index] = scores
            per_view_scores[index] = branch_scores
            labels[index] = label.numpy()
            grades[index] = grade.numpy()
    return fused_scores, per_view_scores, labels, grades


def score_quality(model, loader, device):
    labels, scores = [], []
    model.eval()
    with torch.inference_mode():
        for images, quality, _indices in loader:
            _fused, _views, logits = model(images.to(device, non_blocking=True))
            labels.append(quality.numpy().reshape(-1))
            scores.append(torch.sigmoid(logits.float()).cpu().numpy().reshape(-1))
    y = np.concatenate(labels).astype(np.int64)
    p = np.concatenate(scores)
    predicted = p >= 0.5
    return {
        "n": int(len(y)),
        "adequate": int(y.sum()),
        "inadequate": int(len(y) - y.sum()),
        "auroc": safe_metrics(y, p).get("auroc"),
        "average_precision": safe_metrics(y, p).get("average_precision"),
        "accuracy_at_0_5": float(np.mean(predicted == y)),
        "sensitivity_at_0_5": float(np.mean(predicted[y == 1])) if np.any(y == 1) else None,
        "specificity_at_0_5": float(np.mean(~predicted[y == 0])) if np.any(y == 0) else None,
    }


def write_predictions(path: Path, pairs, fused, branches, grades, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "eye_id", "grade", "referable_dr", "macula_score", "optic_disc_score", "fused_score"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(pairs):
            writer.writerow({
                "patient_id": item["patient_id"],
                "eye_id": item["eye_id"],
                "grade": int(grades[i]),
                "referable_dr": int(labels[i]),
                "macula_score": float(branches[i, 0]),
                "optic_disc_score": float(branches[i, 1]),
                "fused_score": float(fused[i]),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/drtid_untouched_certification/certification_manifest.csv")
    parser.add_argument("--output-dir", default="results/drtid_paired_fusion_dev")
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2, help="Number of eyes; encoder processes two images per eye.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--aux-view-weight", type=float, default=0.25)
    parser.add_argument("--ordinal-weight", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--hard-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=2.0)
    parser.add_argument("--quality-manifest", help="DeepDRiD JSONL manifest with official overall-quality labels.")
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--encoder-lr", type=float, help="Backbone LR; defaults to one tenth of the head LR when fine-tuning all layers.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-full-encoder", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = Path.cwd()
    manifest = Path(args.manifest)
    pairs = read_pairs(manifest, root)
    train_pairs, valid_pairs = patient_split(pairs, args.validation_fraction, args.seed)
    import timm

    model = PairedFusionModel(args.model_name, not args.no_pretrained, not args.train_full_encoder)
    cfg = timm.data.resolve_model_data_config(model.encoder)
    cfg["input_size"] = (3, args.image_size, args.image_size)
    train_transform = timm.data.create_transform(**cfg, is_training=True)
    eval_transform = timm.data.create_transform(**cfg, is_training=False)
    train_ds = PairedFundusDataset(train_pairs, train_transform)
    valid_ds = PairedFundusDataset(valid_pairs, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    quality_train_loader = quality_valid_loader = None
    quality_train_count = quality_valid_count = 0
    q_train_ids: set[str] = set()
    q_valid_ids: set[str] = set()
    quality_manifest_hash = None
    if args.quality_manifest:
        quality_path = Path(args.quality_manifest)
        quality_manifest_hash = hashlib.sha256(quality_path.read_bytes()).hexdigest()
        quality_train_data = DeepDRiDQualityDataset(quality_path, root, train_transform, {"official_train"})
        quality_valid_data = DeepDRiDQualityDataset(quality_path, root, eval_transform, {"official_train"})
        q_train, q_valid = patient_split(quality_train_data.items, args.validation_fraction, args.seed)
        q_train_ids = {str(item["patient_id"]) for item in q_train}
        q_valid_ids = {str(item["patient_id"]) for item in q_valid}
        if q_train_ids & q_valid_ids:
            raise RuntimeError("DeepDRiD quality train/validation patient leakage detected")
        q_train_ds = torch.utils.data.Subset(
            quality_train_data,
            [i for i, item in enumerate(quality_train_data.items) if str(item["patient_id"]) in q_train_ids],
        )
        q_valid_ds = torch.utils.data.Subset(
            quality_valid_data,
            [i for i, item in enumerate(quality_valid_data.items) if str(item["patient_id"]) in q_valid_ids],
        )
        quality_train_count, quality_valid_count = len(q_train_ds), len(q_valid_ds)
        quality_train_loader = DataLoader(q_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        quality_valid_loader = DataLoader(q_valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device(args.device)
    model.to(device)
    named_parameters = list(model.named_parameters())
    head_parameters = [parameter for name, parameter in named_parameters if parameter.requires_grad and not name.startswith("encoder.")]
    layer4_parameters = [parameter for name, parameter in named_parameters if parameter.requires_grad and name.startswith("encoder.layer4.")]
    early_encoder_parameters = [
        parameter for name, parameter in named_parameters
        if parameter.requires_grad and name.startswith("encoder.") and not name.startswith("encoder.layer4.")
    ]
    layer4_lr = args.encoder_lr or args.lr * 0.1
    optimizer_groups = [{"params": head_parameters, "lr": args.lr}]
    if layer4_parameters:
        optimizer_groups.append({"params": layer4_parameters, "lr": layer4_lr})
    if early_encoder_parameters:
        optimizer_groups.append({"params": early_encoder_parameters, "lr": layer4_lr * 0.1})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    history = []
    best = -float("inf")
    best_state = None
    best_path = out / "best_paired_fusion.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        quality_iterator = iter(quality_train_loader) if quality_train_loader is not None else None
        for images, grade, _label, _index in train_loader:
            images = images.to(device, non_blocking=True)
            grade = grade.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fused, branches, _quality_logits = model(images)
                loss = criterion(fused, grade)
                loss = loss + args.aux_view_weight * sum(criterion(branches[:, i], grade) for i in range(2))
            loss = loss + args.ordinal_weight * ordinal_loss(fused, grade)
            loss = loss + args.rank_weight * hard_boundary_rank_loss(
                fused, grade, args.rank_margin, args.hard_top_fraction, args.grade2_rank_weight
            )
            if quality_iterator is not None:
                try:
                    q_images, q_labels, _q_indices = next(quality_iterator)
                except StopIteration:
                    quality_iterator = iter(quality_train_loader)
                    q_images, q_labels, _q_indices = next(quality_iterator)
                q_images = q_images.to(device, non_blocking=True)
                q_labels = q_labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    _q_fused, _q_views, q_logits = model(q_images)
                    q_loss = nn.functional.binary_cross_entropy_with_logits(q_logits.float(), q_labels)
                loss = loss + args.quality_weight * q_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(grade)
            count += len(grade)

        fused, branches, labels, grades = score_model(model, valid_loader, device, len(valid_pairs))
        metrics = safe_metrics(labels, fused)
        metrics["patient_level"] = aggregate_patient(valid_pairs, fused)
        metrics["macula_only_zero_miss"] = safe_metrics(labels, branches[:, 0])
        metrics["optic_disc_only_zero_miss"] = safe_metrics(labels, branches[:, 1])
        record = {"epoch": epoch, "train_loss": total / max(1, count), "validation": metrics}
        if quality_valid_loader is not None:
            record["quality_validation"] = score_quality(model, quality_valid_loader, device)
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        # Select by ranking quality, not by a post-hoc threshold or target score.
        rank_score = float(metrics.get("auroc", -1.0))
        if rank_score > best:
            best = rank_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "model_name": args.model_name,
                "image_size": args.image_size,
                "epoch": epoch,
                "args": vars(args),
                "manifest_sha256": manifest_hash,
                "train_patient_ids": sorted({str(x["patient_id"]) for x in train_pairs}),
                "validation_patient_ids": sorted({str(x["patient_id"]) for x in valid_pairs}),
                "quality_train_patient_ids": sorted(q_train_ids),
                "quality_validation_patient_ids": sorted(q_valid_ids),
                "status": "DEVELOPMENT_CHECKPOINT_NOT_CERTIFIED",
            }, best_path)
        (out / "progress.json").write_text(json.dumps({"status": "IN_PROGRESS", "history": history}, indent=2) + "\n", encoding="utf-8")

    if best_state is None:
        raise RuntimeError("Training produced no ranked checkpoint.")
    model.load_state_dict(best_state)
    final_fused, final_branches, final_labels, final_grades = score_model(model, valid_loader, device, len(valid_pairs))
    write_predictions(out / "validation_predictions.csv", valid_pairs, final_fused, final_branches, final_grades, final_labels)
    summary = {
        "status": "PAIRED_FUSION_DEVELOPMENT_COMPLETE_NOT_CERTIFIED",
        "boundary": "Uses only official DRTiD training patients split into development train/validation. Official testing partition is excluded because it was previously inspected.",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "paired_eyes": len(pairs),
        "train_eyes": len(train_pairs),
        "validation_eyes": len(valid_pairs),
        "train_patients": len({str(x["patient_id"]) for x in train_pairs}),
        "validation_patients": len({str(x["patient_id"]) for x in valid_pairs}),
        "model": args.model_name,
        "encoder_trainable": "layer4 and fusion/view heads" if not args.train_full_encoder else "all encoder layers and heads",
        "learning_rates": {
            "task_heads": args.lr,
            "encoder_layer4": layer4_lr if layer4_parameters else None,
            "encoder_early_layers": layer4_lr * 0.1 if early_encoder_parameters else None,
        },
        "quality_supervision": {
            "manifest": args.quality_manifest,
            "manifest_sha256": quality_manifest_hash,
            "source": "DeepDRiD official_train only; overall quality 1=diagnosable, 0=not diagnosable",
            "weight": args.quality_weight,
            "train_pairs": quality_train_count,
            "validation_pairs": quality_valid_count,
            "validation_patient_overlap": 0,
            "train_patients": len(q_train_ids),
            "validation_patients": len(q_valid_ids),
        } if args.quality_manifest else None,
        "loss": {
            "fused_cross_entropy": 1.0,
            "per_view_auxiliary_cross_entropy": args.aux_view_weight,
            "cumulative_ordinal_bce": args.ordinal_weight,
            "hard_grade1_2_boundary_rank": args.rank_weight,
            "rank_margin": args.rank_margin,
            "hard_pair_fraction": args.hard_top_fraction,
            "grade2_pair_weight": args.grade2_rank_weight,
        },
        "best_checkpoint": str(best_path),
        "history": history,
        "predictions": str(out / "validation_predictions.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
