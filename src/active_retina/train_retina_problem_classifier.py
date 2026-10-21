from __future__ import annotations

import argparse
import ast
import csv
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


LESION_NAMES = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred))) if len(y) else 0.0


def binary_stats(y: np.ndarray, pred: np.ndarray) -> tuple[list[list[int]], float, float]:
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    tn = int(np.sum((y == 0) & (pred == 0))); fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0))); tp = int(np.sum((y == 1) & (pred == 1)))
    tnr = tn / max(tn + fp, 1); tpr = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return [[tn, fp], [fn, tp]], float((tnr + tpr) / 2), float(f1)


def auc_rank(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_records(root: Path, manifest: Path, mmrdr_csv: Path, seed: int) -> list[dict]:
    records = []
    for row in read_rows(manifest):
        split = row["source_split"].rsplit("_", 1)[-1]
        if split not in {"train", "valid", "test"}:
            continue
        grade = int(row["grade"])
        source = row["source_dataset"]
        records.append({
            "path": str((root / row["image_path"]).resolve()),
            "source": source,
            "split": split,
            "grade": grade,
            "binary": int(grade > 0),
            "lesions": [0] * len(LESION_NAMES),
            "lesion_known": False,
            "group": row.get("patient_id", row["image_id"]),
        })

    mm_rows = read_rows(mmrdr_csv)
    train_rows, test_rows = [], []
    for row in mm_rows:
        name = Path(row["image"]).name
        if name.startswith("tr"):
            train_rows.append(row)
        elif name.startswith("ts"):
            test_rows.append(row)
        else:
            raise ValueError(f"Unexpected MMRDR split prefix: {name}")
    rng = np.random.default_rng(seed)
    grades = np.asarray([int(r["grade"]) for r in train_rows])
    tr_idx, va_idx = [], []
    for grade in np.unique(grades):
        indices = np.flatnonzero(grades == grade)
        rng.shuffle(indices)
        n_valid = max(1, int(round(len(indices) * 0.15)))
        va_idx.extend(indices[:n_valid].tolist())
        tr_idx.extend(indices[n_valid:].tolist())
    split_by_index = {int(i): "train" for i in tr_idx}
    split_by_index.update({int(i): "valid" for i in va_idx})
    for split, source_rows in (("train", train_rows), ("test", test_rows)):
        for i, row in enumerate(source_rows):
            grade = int(row["grade"])
            lesions = [int(v) for v in ast.literal_eval(row["lesion"])]
            if len(lesions) != len(LESION_NAMES):
                raise ValueError(f"Expected seven lesion labels, got {lesions}")
            actual_split = split_by_index[i] if split == "train" else "test"
            records.append({
                "path": str((root / "external/MMRDR/MMRDR-UWF" / row["image"]).resolve()),
                "source": "mmrdr_uwf",
                "split": actual_split,
                "grade": grade,
                "binary": int(grade > 0 or any(lesions)),
                "lesions": lesions,
                "lesion_known": True,
                "group": Path(row["image"]).stem,
            })

    missing = [r["path"] for r in records if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images missing; first: {missing[0]}")
    return records


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, train: bool):
        self.rows = rows
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06),
            ]
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["path"]) as im:
            image = self.transform(im.convert("RGB"))
        return (
            image,
            torch.tensor(row["binary"], dtype=torch.float32),
            torch.tensor(row["grade"], dtype=torch.long),
            torch.tensor(row["lesions"], dtype=torch.float32),
            torch.tensor(row["lesion_known"], dtype=torch.bool),
        )


class RetinaModel(nn.Module):
    def __init__(self, name: str, n_grades: int = 6):
        super().__init__()
        if name != "convnext_tiny":
            raise ValueError(f"Unsupported encoder: {name}")
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        width = self.encoder.classifier[-1].in_features
        self.encoder.classifier = nn.Identity()
        self.problem_head = nn.Linear(width, 1)
        self.grade_head = nn.Linear(width, n_grades)
        self.lesion_head = nn.Linear(width, len(LESION_NAMES))

    def forward(self, x):
        z = torch.flatten(self.encoder(x), 1)
        return self.problem_head(z).squeeze(1), self.grade_head(z), self.lesion_head(z)


def metrics(rows: list[dict], probs: np.ndarray, threshold: float, grade_probs: np.ndarray,
            lesion_probs: np.ndarray | None = None, lesion_thresholds: np.ndarray | None = None) -> dict:
    out = {}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["source"]].append(i)
    for source, idx in groups.items():
        y = np.asarray([rows[i]["binary"] for i in idx], dtype=int)
        pred = (probs[idx] >= threshold).astype(int)
        cm, bal_acc, f1 = binary_stats(y, pred)
        item = {
            "n": len(idx), "accuracy": accuracy(y, pred),
            "balanced_accuracy": bal_acc,
            "f1": f1, "confusion_matrix_negpos": cm,
            "positive_prevalence": float(y.mean()),
        }
        if len(np.unique(y)) == 2:
            item["auroc"] = auc_rank(y, probs[idx])
        gtrue = np.asarray([rows[i]["grade"] for i in idx], dtype=int)
        gpred = grade_probs[idx].argmax(axis=1)
        item["grade_accuracy"] = accuracy(gtrue, gpred)
        if lesion_probs is not None and source == "mmrdr_uwf":
            target = np.asarray([rows[i]["lesions"] for i in idx], dtype=int)
            lp = lesion_probs[idx]
            lesion_result = {}
            for j, name in enumerate(LESION_NAMES):
                if len(np.unique(target[:, j])) == 2:
                    auc = auc_rank(target[:, j], lp[:, j])
                else:
                    auc = None
                lesion_result[name] = {
                    "prevalence": float(target[:, j].mean()), "auroc": auc,
                    "accuracy": accuracy(target[:, j], (lp[:, j] >= lesion_thresholds[j]).astype(int)),
                    "threshold": float(lesion_thresholds[j]),
                }
            item["lesions"] = lesion_result
        out[source] = item
    return out


def infer(model, loader, device):
    model.eval()
    ps, gs, ls, ys, gts, lts, knowns = [], [], [], [], [], [], []
    with torch.inference_mode():
        for x, y, grade, lesions, known in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                pl, gl, ll = model(x)
            ps.append(torch.sigmoid(pl).float().cpu().numpy())
            gs.append(torch.softmax(gl.float(), dim=1).cpu().numpy())
            ls.append(torch.sigmoid(ll.float()).cpu().numpy())
            ys.append(y.numpy()); gts.append(grade.numpy()); lts.append(lesions.numpy()); knowns.append(known.numpy())
    return tuple(np.concatenate(x) for x in (ps, gs, ls, ys, gts, lts, knowns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("results/l4_development_combined_manifest.csv"))
    parser.add_argument("--mmrdr-csv", type=Path, default=Path("external/MMRDR/MMRDR-UWF/UWF.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/retina_problem_multitask_l4"))
    parser.add_argument("--artifact", type=Path, default=Path("models/retina_problem_multitask_l4.pt"))
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    root = args.root.resolve()
    records = build_records(root, root / args.manifest, root / args.mmrdr_csv, args.seed)
    split_rows = {s: [r for r in records if r["split"] == s] for s in ("train", "valid", "test")}
    for s, rows in split_rows.items():
        print(json.dumps({"split": s, "n": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
                          "binary_counts": dict(Counter(str(r["binary"]) for r in rows))}), flush=True)

    train_ds = FundusDataset(split_rows["train"], args.image_size, True)
    val_ds = FundusDataset(split_rows["valid"], args.image_size, False)
    test_ds = FundusDataset(split_rows["test"], args.image_size, False)
    src_counts = Counter(r["source"] for r in split_rows["train"])
    sample_weights = [1.0 / src_counts[r["source"]] for r in split_rows["train"]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True,
                   "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This run was requested for L4; CUDA is not available")
    model = RetinaModel(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    binary_loss = nn.BCEWithLogitsLoss()
    grade_counts = np.bincount([r["grade"] for r in split_rows["train"]], minlength=6).astype(np.float32)
    grade_weights = np.sqrt(grade_counts.sum() / np.maximum(grade_counts, 1.0))
    grade_weights = grade_weights / grade_weights.mean()
    grade_loss = nn.CrossEntropyLoss(weight=torch.tensor(grade_weights, dtype=torch.float32, device=device))
    mm_train = [r for r in split_rows["train"] if r["lesion_known"]]
    lesion_counts = np.asarray([r["lesions"] for r in mm_train], dtype=np.float32).sum(axis=0)
    lesion_pos_weight = (len(mm_train) - lesion_counts) / np.maximum(lesion_counts, 1)
    lesion_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.minimum(lesion_pos_weight, 20), dtype=torch.float32, device=device))
    args.outdir = root / args.outdir; args.outdir.mkdir(parents=True, exist_ok=True)
    args.artifact = root / args.artifact; args.artifact.parent.mkdir(parents=True, exist_ok=True)
    config = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sources": dict(Counter(r["source"] for r in records)), "labels": LESION_NAMES,
              "target_definition": "requires_attention = DR grade > 0 OR any MMRDR-UWF annotated lesion; DDR grade 5 ungradable is flagged as requiring attention",
              "split_caveat": "Source test partitions are development diagnostics with prior project exposure; MMRDR lacks patient identifiers.",
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2))
    best_acc, stale, history = -1.0, 0, []
    best_epoch = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y, grade, lesions, known in train_loader:
            x=x.to(device, non_blocking=True); y=y.to(device); grade=grade.to(device); lesions=lesions.to(device); known=known.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pl, gl, ll = model(x)
                loss = binary_loss(pl, y) + 0.45 * grade_loss(gl, grade)
                if known.any():
                    loss = loss + 0.35 * lesion_loss(ll[known], lesions[known])
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        p, gp, lp, yv, gv, lv, kv = infer(model, val_loader, device)
        candidate_thresholds = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 193)))
        accs = [accuracy(yv.astype(int), (p >= t).astype(int)) for t in candidate_thresholds]
        best_idx = int(np.argmax(accs)); threshold, val_acc = float(candidate_thresholds[best_idx]), float(accs[best_idx])
        epoch_item = {"epoch": epoch, "loss": float(np.mean(losses)), "val_accuracy": val_acc,
                      "val_threshold": threshold, "val_metrics": metrics(split_rows["valid"], p, threshold, gp)}
        history.append(epoch_item)
        print(json.dumps(epoch_item), flush=True)
        if val_acc > best_acc:
            best_acc, stale, best_epoch = val_acc, 0, epoch
            # Select lesion thresholds on labeled validation MMRDR rows only.
            mm_idx = np.flatnonzero(kv.astype(bool))
            yles = lv[mm_idx].astype(int); ples = lp[mm_idx]
            lesion_thresholds = np.asarray([
                max(np.unique(np.quantile(ples[:, j], np.linspace(0.01, 0.99, 99))),
                    key=lambda t: accuracy(yles[:, j], (ples[:, j] >= t).astype(int)))
                for j in range(len(LESION_NAMES))
            ])
            torch.save({"state_dict": model.state_dict(), "model": args.model, "image_size": args.image_size,
                        "problem_threshold": threshold, "lesion_thresholds": lesion_thresholds.tolist(),
                        "lesion_names": LESION_NAMES, "epoch": epoch, "config": config}, args.artifact)
        else:
            stale += 1
            if stale >= args.patience:
                print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
                break
    (args.outdir / "history.json").write_text(json.dumps(history, indent=2))

    checkpoint = torch.load(args.artifact, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    threshold = float(checkpoint["problem_threshold"])
    lesion_thresholds = np.asarray(checkpoint["lesion_thresholds"])
    p, gp, lp, yt, gt, lt, kt = infer(model, test_loader, device)
    report = {
        "status": "DEVELOPMENT_EVALUATION_NOT_CLINICAL_VALIDATION",
        "selected_epoch": checkpoint["epoch"], "problem_threshold": threshold,
        "test_overall": metrics(split_rows["test"], p, threshold, gp, lp, lesion_thresholds),
        "n_test": len(split_rows["test"]), "elapsed_seconds": time.time() - started,
        "split_caveat": config["split_caveat"],
    }
    (args.outdir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    with (args.outdir / "test_predictions.csv").open("w", newline="") as f:
        writer=csv.writer(f); writer.writerow(["path","source","grade_true","problem_true","problem_probability","problem_prediction",*([f"p_{n}" for n in LESION_NAMES]),*([f"lesion_{n}_true" for n in LESION_NAMES])])
        for i, r in enumerate(split_rows["test"]):
            writer.writerow([r["path"],r["source"],r["grade"],r["binary"],float(p[i]),int(p[i]>=threshold),*lp[i].tolist(),*r["lesions"]])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
