"""Train a selective nonreferable-clearance ranker on cached DDR features.

The model is deliberately optimized for the autonomous-use failure mode:
clear as many nonreferable images as possible while keeping referable images
out of the clearance region.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .formal_risk_control import binomial_upper


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def load_split(rows: list[dict[str, Any]], features: dict[str, np.ndarray], split: str):
    selected = [r for r in rows if r["source_split"] == f"ddr_{split}" and r["image_id"] in features]
    x = np.vstack([features[r["image_id"]] for r in selected]).astype("float32")
    y = np.asarray([r["referable_dr"] for r in selected], dtype="float32")
    grade = np.asarray([r["grade"] for r in selected], dtype="int64")
    ids = [r["image_id"] for r in selected]
    return x, y, grade, ids


class ClearanceRanker(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, n_grades: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden // 2, 1)
        self.grade_head = nn.Linear(hidden // 2, n_grades)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.risk_head(z).flatten(), self.grade_head(z)


def standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, *((x - mean) / std for x in others), mean.astype("float32"), std.astype("float32")


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    risks: list[np.ndarray] = []
    grade_probs: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (xb,) in loader:
            logits, grade_logits = model(xb.to(device))
            risks.append(torch.sigmoid(logits).detach().cpu().numpy())
            grade_probs.append(torch.softmax(grade_logits, dim=1).detach().cpu().numpy())
    return np.concatenate(risks).astype(float), np.vstack(grade_probs).astype(float)


def asymmetric_focal_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: float, gamma_pos: float, gamma_neg: float) -> torch.Tensor:
    """Focal BCE that keeps referable misses expensive while mining hard negatives."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        y,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1.0 - p)
    gamma = torch.where(y == 1, torch.tensor(gamma_pos, device=logits.device), torch.tensor(gamma_neg, device=logits.device))
    return (((1.0 - pt).clamp_min(1e-6) ** gamma) * bce).mean()


def hard_mining_weights(
    y: np.ndarray,
    grade: np.ndarray,
    scores: np.ndarray | None,
    pos_weight: float,
    grade2_boost: float,
    grade1_boost: float,
    hard_fraction: float,
    hard_boost: float,
) -> np.ndarray:
    weights = np.ones_like(y, dtype="float32")
    weights[y == 1] = pos_weight
    weights[(y == 1) & (grade == 2)] *= grade2_boost
    weights[(y == 0) & (grade == 1)] *= grade1_boost
    if scores is None or hard_fraction <= 0:
        return weights
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    if len(positives):
        n = max(1, int(np.ceil(len(positives) * hard_fraction)))
        # Dangerous positives are the ones with the lowest risk score.
        hard_pos = positives[np.argsort(scores[positives])[:n]]
        weights[hard_pos] *= hard_boost
    if len(negatives):
        n = max(1, int(np.ceil(len(negatives) * hard_fraction)))
        # Useful hard negatives are the ones the model currently refuses to clear.
        hard_neg = negatives[np.argsort(scores[negatives])[-n:]]
        weights[hard_neg] *= hard_boost
    return weights


def clearance_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    clear = scores <= threshold
    positives = y == 1
    negatives = y == 0
    false_clear = int((positives & clear).sum())
    true_clear = int((negatives & clear).sum())
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    ucb = binomial_upper(false_clear, n_pos - false_clear, 0.05) if n_pos else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": n_pos,
        "negative": n_neg,
        "true_nonreferable_cleared": true_clear,
        "false_nonreferable_clearance": false_clear,
        "nonreferable_clearance_specificity": float(true_clear / max(1, n_neg)),
        "clearance_coverage": float(clear.mean()),
        "missed_referable_rate": float(false_clear / max(1, n_pos)),
        "missed_risk_ucb95": ucb,
        "zero_miss": false_clear == 0,
    }


def select_threshold_zero_miss(y: np.ndarray, scores: np.ndarray, safety_margin: float) -> dict[str, Any]:
    positive_scores = scores[y == 1]
    if len(positive_scores) == 0:
        threshold = float(np.max(scores))
    else:
        threshold = max(float(np.min(positive_scores)) - safety_margin, float(np.min(scores)) - 1e-8)
    return clearance_metrics(y, scores, threshold)


def pairwise_margin_loss(logits: torch.Tensor, y: torch.Tensor, grade: torch.Tensor, margin: float) -> torch.Tensor:
    pos = logits[y == 1]
    neg = logits[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    pos_grade = grade[y == 1].float()
    # Grade-2 cases sit exactly on the referable boundary; missing them is the
    # dangerous failure, so they get extra pairwise pressure.
    pos_weight = torch.where(pos_grade <= 2, torch.tensor(2.5, device=logits.device), torch.tensor(1.0, device=logits.device))
    losses = torch.relu(margin - (pos[:, None] - neg[None, :]))
    return (losses * pos_weight[:, None]).mean()


def write_predictions(path: Path, split: str, ids: list[str], y: np.ndarray, grade: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_id", "grade", "referable_dr", "risk_score", "action"],
        )
        if not exists:
            writer.writeheader()
        for image_id, yy, gg, score in zip(ids, y, grade, scores):
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "grade": int(gg),
                    "referable_dr": int(yy),
                    "risk_score": f"{float(score):.8f}",
                    "action": "AUTONOMOUS_NONREFERABLE" if score <= threshold else "DEFER_OR_REFER",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--pos-weight", type=float, default=12.0)
    parser.add_argument("--grade2-boost", type=float, default=3.0)
    parser.add_argument("--grade1-boost", type=float, default=3.0)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--hard-boost", type=float, default=8.0)
    parser.add_argument("--hard-refresh-epochs", type=int, default=10)
    parser.add_argument("--rank-weight", type=float, default=3.0)
    parser.add_argument("--grade-weight", type=float, default=0.2)
    parser.add_argument("--focal-gamma-pos", type=float, default=2.0)
    parser.add_argument("--focal-gamma-neg", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--safety-margin", type=float, default=1e-6)
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    rows = read_manifest(Path(args.manifest))
    raw = np.load(args.feature_cache, allow_pickle=False)
    features = {k: raw[k].astype("float32") for k in raw.files}
    x_train, y_train, g_train, _ = load_split(rows, features, "train")
    x_valid, y_valid, g_valid, valid_ids = load_split(rows, features, "valid")
    x_test, y_test, g_test, test_ids = load_split(rows, features, "test")
    x_train, x_valid, x_test, mean, std = standardize(x_train, x_valid, x_test)

    sample_weight = hard_mining_weights(
        y_train,
        g_train,
        None,
        args.pos_weight,
        args.grade2_boost,
        args.grade1_boost,
        args.hard_fraction,
        args.hard_boost,
    )

    model = ClearanceRanker(x_train.shape[1], args.hidden, args.dropout).to(device)
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_epoch: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or (args.hard_refresh_epochs > 0 and (epoch - 1) % args.hard_refresh_epochs == 0):
            if epoch > 1:
                train_scores, _ = predict(model, x_train, args.batch_size, device)
                sample_weight = hard_mining_weights(
                    y_train,
                    g_train,
                    train_scores,
                    args.pos_weight,
                    args.grade2_boost,
                    args.grade1_boost,
                    args.hard_fraction,
                    args.hard_boost,
                )
            sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(g_train)),
                batch_size=args.batch_size,
                sampler=sampler,
            )
        model.train()
        total = 0.0
        seen = 0
        for xb, yb, gb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            gb = gb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, grade_logits = model(xb)
            loss = asymmetric_focal_bce(
                logits,
                yb,
                args.pos_weight,
                args.focal_gamma_pos,
                args.focal_gamma_neg,
            )
            loss = loss + args.rank_weight * pairwise_margin_loss(logits, yb, gb, args.margin)
            loss = loss + args.grade_weight * ce(grade_logits, gb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(yb)
            seen += len(yb)
        valid_scores, _ = predict(model, x_valid, args.batch_size, device)
        valid_row = select_threshold_zero_miss(y_valid, valid_scores, args.safety_margin)
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "valid_clearance": valid_row}
        history.append(row)
        if best_epoch is None or (
            int(valid_row["zero_miss"]),
            valid_row["nonreferable_clearance_specificity"],
            valid_row["clearance_coverage"],
            -valid_row["missed_risk_ucb95"],
        ) > (
            int(best_epoch["valid_clearance"]["zero_miss"]),
            best_epoch["valid_clearance"]["nonreferable_clearance_specificity"],
            best_epoch["valid_clearance"]["clearance_coverage"],
            -best_epoch["valid_clearance"]["missed_risk_ucb95"],
        ):
            best_epoch = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or valid_row["nonreferable_clearance_specificity"] >= args.target_specificity:
            print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_scores, valid_grade_probs = predict(model, x_valid, args.batch_size, device)
    selected = select_threshold_zero_miss(y_valid, valid_scores, args.safety_margin)
    threshold = float(selected["threshold"])
    test_scores, test_grade_probs = predict(model, x_test, args.batch_size, device)
    valid_metrics = clearance_metrics(y_valid, valid_scores, threshold)
    test_metrics = clearance_metrics(y_test, test_scores, threshold)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid_ids, y_valid, g_valid, valid_scores, threshold)
    write_predictions(pred_path, "test", test_ids, y_test, g_test, test_scores, threshold)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "threshold": threshold,
            "mean": mean,
            "std": std,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        args.artifact,
    )
    summary = {
        "status": (
            "GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS"
            if test_metrics["zero_miss"] and test_metrics["nonreferable_clearance_specificity"] >= args.target_specificity
            else "NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS"
        ),
        "manifest": args.manifest,
        "feature_cache": args.feature_cache,
        "artifact": args.artifact,
        "best_epoch": best_epoch,
        "threshold_selected_on_validation": threshold,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "target_specificity": args.target_specificity,
        "predictions": str(pred_path),
        "valid_grade_probability_shape": list(valid_grade_probs.shape),
        "test_grade_probability_shape": list(test_grade_probs.shape),
        "history_tail": history[-20:],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
