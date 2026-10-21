from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


LESIONS = ("MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD")
TASKS = ("dr", "quality", "lesion", "gon")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stratified_patient_splits(rows: list[dict], seed: int) -> dict[str, str]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    for positive in (False, True):
        patients = [p for p, items in by_patient.items()
                    if any(item["gon"] == 1 for item in items) == positive]
        rng.shuffle(patients)
        n = len(patients)
        n_valid = max(1, round(n * 0.15)) if n >= 8 else 0
        n_test = max(1, round(n * 0.15)) if n >= 8 else 0
        for index, patient in enumerate(patients):
            if index < n_test:
                split_map[patient] = "test"
            elif index < n_test + n_valid:
                split_map[patient] = "valid"
            else:
                split_map[patient] = "train"
    return split_map


def build_rows(root: Path, manifest: Path, mmrdr_manifest: Path,
               mmrdr_labels: Path, hyg_dataset: Path, seed: int,
               include_mmrdr: bool = True) -> list[dict]:
    rows: list[dict] = []
    for item in read_csv(root / manifest):
        grade = int(item["grade"])
        split = item["source_split"].rsplit("_", 1)[-1]
        if split not in {"train", "valid", "test"}:
            continue
        if grade not in range(6):
            raise ValueError(f"Unexpected DR grade {grade}: {item}")
        rows.append({
            "image_path": str((root / item["image_path"]).resolve()),
            "source": item["source_dataset"], "split": split,
            "patient_id": item.get("patient_id") or item["image_id"],
            "patient_group_verified": item["source_dataset"] in {"drtid_paired", "deepdrid"},
            "image_id": item["image_id"],
            "dr_grade": grade if grade < 5 else -1,
            "ungradable": int(grade == 5) if item["source_dataset"] == "ddr_hf_specificity300" else -1,
            "lesions": [-1] * len(LESIONS), "gon": -1,
        })

    if include_mmrdr:
        lesion_by_image = {Path(row["image"]).name: [int(v) for v in row["lesion"].strip("[]").replace(" ", "").split(",")]
                           for row in read_csv(root / mmrdr_labels)}
        for item in read_csv(root / mmrdr_manifest):
            name = Path(item["image_path"]).name
            labels = lesion_by_image.get(name)
            if labels is None or len(labels) != len(LESIONS):
                raise ValueError(f"Missing/invalid MMRDR lesion vector for {name}")
            grade = int(item["grade"])
            rows.append({
                "image_path": str((root / item["image_path"]).resolve()),
                "source": "mmrdr_uwf", "split": item["source_split"],
                "patient_id": item.get("patient_id") or f"mmrdr-image:{name}",
                "patient_group_verified": False,
                "image_id": item["image_id"], "dr_grade": grade if grade < 5 else -1,
                "ungradable": -1, "lesions": labels, "gon": -1,
            })

    hyg = []
    for item in read_csv(root / hyg_dataset / "Labels.csv"):
        label = item["Label"].strip()
        if label not in {"GON+", "GON-"}:
            raise ValueError(f"Unexpected HYGD label: {label}")
        hyg.append({
            "image_path": str((root / hyg_dataset / "Images" / item["Image Name"]).resolve()),
            "source": "hygd_v1_1_0", "split": "", "patient_id": item["Patient"],
            "patient_group_verified": True,
            "image_id": item["Image Name"], "dr_grade": -1, "ungradable": -1,
            "lesions": [-1] * len(LESIONS), "gon": int(label == "GON+"),
        })
    hyg_splits = stratified_patient_splits(hyg, seed)
    for item in hyg:
        item["split"] = hyg_splits[item["patient_id"]]
    rows.extend(hyg)

    missing = [row["image_path"] for row in rows if not Path(row["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[0]}")
    return rows


def deduplicate_exact_images(rows: list[dict]) -> tuple[list[dict], dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        row["image_sha256"] = digest
        by_hash[digest].append(row)
    kept, removed, cross_split, conflicting_groups = [], 0, 0, 0
    split_order = {"train": 0, "valid": 1, "test": 2}
    for digest, copies in by_hash.items():
        task_values = defaultdict(set)
        for row in copies:
            if row["dr_grade"] >= 0: task_values["dr"].add(row["dr_grade"])
            if row["ungradable"] >= 0: task_values["quality"].add(row["ungradable"])
            if row["gon"] >= 0: task_values["gon"].add(row["gon"])
            for j, value in enumerate(row["lesions"]):
                if value >= 0: task_values[f"lesion_{j}"].add(value)
        if any(len(values) > 1 for values in task_values.values()):
            conflicting_groups += 1
            removed += len(copies)
            continue
        splits = {row["split"] for row in copies}
        if len(splits) > 1:
            cross_split += 1
            preferred = min(splits, key=lambda value: split_order[value])
            copies = [row for row in copies if row["split"] == preferred]
        source_seen = set()
        for row in sorted(copies, key=lambda item: (item["source"], item["image_id"])):
            key = (row["source"], digest)
            if key in source_seen:
                removed += 1
                continue
            source_seen.add(key)
            kept.append(row)
    return kept, {"unique_sha256_images": len(by_hash), "cross_split_duplicate_groups": cross_split,
                  "conflicting_label_groups_quarantined": conflicting_groups,
                  "rows_removed": len(rows) - len(kept),
                  "same_source_repeat_rows_removed": removed,
                  "rows_before": len(rows), "rows_after": len(kept)}


class RetinaRows(Dataset):
    def __init__(self, rows: list[dict], size: int, train: bool):
        self.rows = rows
        ops = [transforms.Resize((size, size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            x = self.transform(image.convert("RGB"))
        lesions = row["lesions"]
        return (x, torch.tensor(row["dr_grade"]), torch.tensor(row["ungradable"]),
                torch.tensor(lesions, dtype=torch.float32), torch.tensor(row["gon"]))


class ScopedRetinaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        width = self.encoder.classifier[-1].in_features
        self.encoder.classifier[-1] = nn.Identity()
        self.dr_grade = nn.Linear(width, 5)
        self.quality = nn.Linear(width, 1)
        self.mmrdr_lesions = nn.Linear(width, len(LESIONS))
        self.glaucoma = nn.Linear(width, 1)

    def forward(self, x):
        z = torch.flatten(self.encoder(x), 1)
        return self.dr_grade(z), self.quality(z).squeeze(1), self.mmrdr_lesions(z), self.glaucoma(z).squeeze(1)


class BoundedTaskWeights(nn.Module):
    """Learn modest homoscedastic task weights, bounded to [0.5, 2]."""
    def __init__(self):
        super().__init__()
        self.log_variance = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in TASKS})

    def combine(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [torch.exp(-self.log_variance[name]) * loss + self.log_variance[name]
                 for name, loss in task_losses.items()]
        return torch.stack(terms).mean()

    def clamp_(self) -> None:
        with torch.no_grad():
            for value in self.log_variance.values():
                value.clamp_(min=-np.log(2.0), max=np.log(2.0))

    def weights(self) -> dict[str, float]:
        return {name: float(torch.exp(-value.detach()).cpu())
                for name, value in self.log_variance.items()}


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        return None
    pos, neg = score[y == 1], score[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def summarize(rows: list[dict], outputs: tuple[np.ndarray, ...]) -> dict:
    dr, quality, lesions, gon = outputs
    reports = {}
    sources = sorted({row["source"] for row in rows})
    for source in sources:
        source_idx = [i for i, row in enumerate(rows) if row["source"] == source]
        scoped = {}
        for task, mask_key, score, threshold in (
            ("ungradable", "ungradable", quality, 0.5),
            ("glaucoma", "gon", gon, 0.5),
        ):
            idx = [i for i in source_idx if rows[i][mask_key] >= 0]
            if not idx:
                continue
            y = np.asarray([rows[i][mask_key] for i in idx], dtype=int)
            pred = (score[idx] >= threshold).astype(int)
            tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
            scoped[task] = {"n_samples": len(idx),
                            "n_unique_patient_ids": len({rows[i]["patient_id"] for i in idx}),
                            "patient_ids_verified": all(rows[i]["patient_group_verified"] for i in idx),
                            "accuracy_at_0_5": float((pred == y).mean()), "auroc": safe_auc(y, score[idx]),
                            "sensitivity_at_0_5": tp / max(tp + fn, 1), "specificity_at_0_5": tn / max(tn + fp, 1),
                            "confusion_matrix": [[tn, fp], [fn, tp]]}
        dr_idx = [i for i in source_idx if rows[i]["dr_grade"] >= 0]
        if dr_idx:
            yt = np.asarray([rows[i]["dr_grade"] for i in dr_idx], dtype=int)
            yp = dr[dr_idx].argmax(axis=1)
            referable_true = yt >= 2
            referable_score = np.exp(dr[dr_idx] - dr[dr_idx].max(axis=1, keepdims=True))
            referable_score = referable_score[:, 2:].sum(axis=1) / referable_score.sum(axis=1)
            referable_pred = referable_score >= 0.5
            scoped["dr_grade_0_to_4"] = {
                "n_samples": len(dr_idx),
                "n_unique_patient_ids": len({rows[i]["patient_id"] for i in dr_idx}),
                "patient_ids_verified": all(rows[i]["patient_group_verified"] for i in dr_idx),
                "accuracy": float((yt == yp).mean()),
                "macro_recall": float(np.mean([np.mean(yp[yt == c] == c) for c in np.unique(yt)])),
                "labels_present": sorted(np.unique(yt).tolist())}
            scoped["referable_dr_binary_grade_ge_2"] = {
                "n_samples": len(dr_idx),
                "positive": int(referable_true.sum()),
                "accuracy_at_0_5": float((referable_pred == referable_true).mean()),
                "auroc": safe_auc(referable_true.astype(int), referable_score),
                "patient_ids_verified": all(rows[i]["patient_group_verified"] for i in dr_idx),
                "threshold": 0.5,
            }
        reports[source] = scoped
    lesion_idx = [i for i, row in enumerate(rows) if any(v >= 0 for v in row["lesions"])]
    if lesion_idx:
        out = {}
        for j, name in enumerate(LESIONS):
            valid = [i for i in lesion_idx if rows[i]["lesions"][j] >= 0]
            y = np.asarray([rows[i]["lesions"][j] for i in valid], dtype=int)
            s = 1 / (1 + np.exp(-lesions[valid, j]))
            out[name] = {"n": len(valid), "positive": int(y.sum()), "auroc": safe_auc(y, s)}
        reports["mmrdr_lesions"] = out
    return reports


def infer(model, loader, device):
    model.eval(); collected = [[] for _ in range(4)]
    with torch.inference_mode():
        for x, *_ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(x)
            for dest, values in zip(collected, outputs):
                dest.extend(values.float().cpu().numpy())
    return (np.asarray(collected[0]), np.asarray(collected[1]),
            np.asarray(collected[2]), np.asarray(collected[3]))


def validation_loss(model, loader, device, losses) -> float:
    dr_loss, quality_loss, lesion_loss, gon_loss = losses
    task_values = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for x, grade, ungradable, lesions, gon in loader:
            x=x.to(device, non_blocking=True); grade=grade.to(device); ungradable=ungradable.to(device)
            lesions=lesions.to(device); gon=gon.to(device)
            dr_logits, q_logits, lesion_logits, gon_logits = model(x)
            masks = {
                "dr": grade >= 0,
                "quality": ungradable >= 0,
                "lesion": (lesions >= 0).all(dim=1),
                "gon": gon >= 0,
            }
            if masks["dr"].any(): task_values["dr"].append(float(dr_loss(dr_logits[masks["dr"]], grade[masks["dr"]])))
            if masks["quality"].any(): task_values["quality"].append(float(quality_loss(q_logits[masks["quality"]], ungradable[masks["quality"]].float())))
            if masks["lesion"].any(): task_values["lesion"].append(float(lesion_loss(lesion_logits[masks["lesion"]], lesions[masks["lesion"]])))
            if masks["gon"].any(): task_values["gon"].append(float(gon_loss(gon_logits[masks["gon"]], gon[masks["gon"]].float())))
    means = [float(np.mean(values)) for values in task_values.values() if values]
    return float(np.mean(means)) if means else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Train retinal heads with source-specific masked labels.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("results/l4_development_combined_manifest.csv"))
    parser.add_argument("--mmrdr-manifest", type=Path, default=Path("results/mmrdr_uwf_manifest_20261013.csv"))
    parser.add_argument("--mmrdr-labels", type=Path, default=Path("external/MMRDR/MMRDR-UWF/UWF.csv"))
    parser.add_argument("--hygd", type=Path, default=Path("external/HYGD_v1_1_0"))
    parser.add_argument("--outdir", type=Path, default=Path("results/scoped_multitask_l4"))
    parser.add_argument("--artifact", type=Path, default=Path("models/scoped_multitask_l4.pt"))
    parser.add_argument("--size", type=int, default=320); parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5); parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Optional model checkpoint for a fine-tuning stage.")
    parser.add_argument("--adaptive-task-weighting", action="store_true",
                        help="Learn bounded task-loss weights from training losses (not reinforcement learning).")
    parser.add_argument("--max-seconds", type=int, default=6 * 60 * 60,
                        help="Hard wall-clock limit; checked between batches and epochs.")
    parser.add_argument("--skip-mmrdr", action="store_true",
                        help="Skip MMRDR-UWF rows when its large image archive is not staged.")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    root = args.root.resolve()
    rows = build_rows(root, args.manifest, args.mmrdr_manifest, args.mmrdr_labels,
                      args.hygd, args.seed, include_mmrdr=not args.skip_mmrdr)
    rows, duplicate_audit = deduplicate_exact_images(rows)
    split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    for split, selected in split_rows.items():
        print(json.dumps({"split": split, "n": len(selected), "sources": dict(Counter(r["source"] for r in selected)),
                          "dr_known": sum(r["dr_grade"] >= 0 for r in selected),
                          "quality_known": sum(r["ungradable"] >= 0 for r in selected),
                          "gon_known": sum(r["gon"] >= 0 for r in selected),
                          "lesion_known": sum(any(v >= 0 for v in r["lesions"]) for r in selected)}), flush=True)
    # Persist the exact split and row provenance before fitting.
    outdir = root / args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    csv_write(outdir / "split_manifest.csv", rows)
    digest = hashlib.sha256((outdir / "split_manifest.csv").read_bytes()).hexdigest()
    (outdir / "duplicate_audit.json").write_text(json.dumps(duplicate_audit, indent=2))
    print(json.dumps({"exact_duplicate_audit": duplicate_audit}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This six-hour run requires the L4 CUDA device")
    model = ScopedRetinaNet().to(device)
    if args.init_checkpoint:
        initial = torch.load(root / args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["state_dict"])
    task_weights = BoundedTaskWeights().to(device) if args.adaptive_task_weighting else None
    optimizer_params = list(model.parameters())
    if task_weights is not None:
        optimizer_params += list(task_weights.parameters())
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=2e-4)
    class_counts = np.bincount([r["dr_grade"] for r in split_rows["train"] if r["dr_grade"] >= 0], minlength=5)
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1)); class_weights /= class_weights.mean()
    dr_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))
    quality_loss = nn.BCEWithLogitsLoss()
    lesion_loss = nn.BCEWithLogitsLoss()
    gon_loss = nn.BCEWithLogitsLoss()
    train_ds = RetinaRows(split_rows["train"], args.size, True)
    counts = Counter(r["source"] for r in split_rows["train"])
    sampler = WeightedRandomSampler([1 / counts[r["source"]] for r in split_rows["train"]],
                                    len(split_rows["train"]), replacement=True)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True,
                     "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    valid_loader = DataLoader(RetinaRows(split_rows["valid"], args.size, False), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(RetinaRows(split_rows["test"], args.size, False), shuffle=False, **loader_kwargs)
    scaler = torch.amp.GradScaler("cuda"); history = []; best = float("inf"); stale = 0
    started = time.time()
    timed_out = False
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_losses = []
        for x, grade, ungradable, lesions, gon in train_loader:
            if time.time() - started >= args.max_seconds:
                timed_out = True
                break
            x=x.to(device, non_blocking=True); grade=grade.to(device); ungradable=ungradable.to(device)
            lesions=lesions.to(device); gon=gon.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                dr_logits, q_logits, lesion_logits, gon_logits = model(x)
                pieces = {}
                dr_known = grade >= 0
                if dr_known.any(): pieces["dr"] = dr_loss(dr_logits[dr_known], grade[dr_known])
                q_known = ungradable >= 0
                if q_known.any(): pieces["quality"] = quality_loss(q_logits[q_known], ungradable[q_known].float())
                lesion_known = (lesions >= 0).all(dim=1)
                if lesion_known.any(): pieces["lesion"] = lesion_loss(lesion_logits[lesion_known], lesions[lesion_known])
                gon_known = gon >= 0
                if gon_known.any(): pieces["gon"] = gon_loss(gon_logits[gon_known], gon[gon_known].float())
                if not pieces: continue
                loss = task_weights.combine(pieces) if task_weights is not None else torch.stack(list(pieces.values())).mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if task_weights is not None:
                task_weights.clamp_()
            epoch_losses.append(float(loss.detach().cpu()))
        if timed_out and not epoch_losses:
            break
        if not epoch_losses:
            raise RuntimeError("No training batches produced a supervised loss")
        valid_outputs = infer(model, valid_loader, device)
        # Use aggregate task loss for early stopping without selecting thresholds on test.
        val_loss = validation_loss(model, valid_loader, device,
                                   (dr_loss, quality_loss, lesion_loss, gon_loss))
        item = {"epoch": epoch, "validation_loss": val_loss,
                "adaptive_task_weights": task_weights.weights() if task_weights is not None else None,
                "validation_metrics": summarize(split_rows["valid"], valid_outputs)}
        history.append(item); print(json.dumps(item), flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"state_dict": model.state_dict(),
                        "task_log_variance": ({name: float(value.detach().cpu())
                                               for name, value in task_weights.log_variance.items()}
                                              if task_weights is not None else None),
                        "epoch": epoch, "args": vars(args),
                        "split_sha256": digest, "heads": ["dr_grade_0_4", "ungradable", "mmrdr_7_lesions", "hygd_gon"]},
                       root / args.artifact)
        else:
            stale += 1
            if stale >= 5: break
        if timed_out or time.time() - started >= args.max_seconds:
            timed_out = True
            break
    if not (root / args.artifact).is_file():
        raise RuntimeError("No checkpoint was saved before the time limit")
    checkpoint = torch.load(root / args.artifact, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_outputs = infer(model, test_loader, device)
    report = {"status": "SCOPED_DEVELOPMENT_ONLY_NOT_CLINICAL_VALIDATION",
              "test_metrics": summarize(split_rows["test"], test_outputs),
              "best_epoch": checkpoint["epoch"], "split_manifest_sha256": digest,
              "duplicate_audit": duplicate_audit,
              "elapsed_seconds": time.time()-started,
              "time_limit_reached": timed_out,
              "max_seconds": args.max_seconds,
              "trained_heads": ["dr_grade_0_4", "ungradable", "hygd_gon"],
              "not_trained_heads": ["mmrdr_7_lesions"] if args.skip_mmrdr else [],
              "global_any_disease_binary": "NOT_TRAINED_INVALID_WITH_PARTIAL_SOURCE_LABELS",
              "label_scope": "Each head is scored only where its source supplied that label; HYGD split is patient-disjoint. Existing data were previously inspected/exposed.",
              "history": history}
    report["adaptive_task_weights"] = task_weights.weights() if task_weights is not None else None
    (outdir / "run_summary.json").write_text(json.dumps(report, indent=2, default=str))
    dr, quality, lesions, gon = test_outputs
    with (outdir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "split", "image_id", "patient_id", "patient_group_verified",
                         "dr_grade_true", "dr_grade_pred", "quality_ungradable_true", "quality_probability",
                         "gon_true", "gon_probability", *[f"lesion_true_{name}" for name in LESIONS],
                         *[f"lesion_probability_{name}" for name in LESIONS]])
        for i, row in enumerate(split_rows["test"]):
            writer.writerow([row["source"], row["split"], row["image_id"], row["patient_id"],
                             row["patient_group_verified"], row["dr_grade"], int(dr[i].argmax()),
                             row["ungradable"], float(1 / (1 + np.exp(-quality[i]))), row["gon"],
                             float(1 / (1 + np.exp(-gon[i]))), *row["lesions"],
                             *[float(1 / (1 + np.exp(-v))) for v in lesions[i]]])
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
