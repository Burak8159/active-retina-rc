"""Train nonlinear MLP heads on cached DDR feature vectors."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .formal_risk_control import binomial_upper


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["grade"] = int(row["grade"])
            row["referable_dr"] = int(row["referable_dr"])
            rows.append(row)
    return rows


def load_xy(rows: list[dict[str, Any]], features: dict[str, np.ndarray], split: str):
    selected = [r for r in rows if r["source_split"] == f"ddr_{split}" and r["image_id"] in features]
    x = np.vstack([features[r["image_id"]] for r in selected]).astype("float32")
    y = np.asarray([int(r["referable_dr"]) for r in selected], dtype="float32")
    ids = [r["image_id"] for r in selected]
    return x, y, ids


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device))
            out.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(out).astype(float)


def eval_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    decided = scores >= threshold
    decided_pos = int(((y == 1) & decided).sum())
    missed = int(((y == 1) & (~decided)).sum())
    false_pos = int(((y == 0) & decided).sum())
    true_neg = int(((y == 0) & (~decided)).sum())
    ucb = binomial_upper(missed, decided_pos, 0.05) if decided_pos else float("nan")
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "decided": int(decided.sum()),
        "decided_positive": decided_pos,
        "missed_positive": missed,
        "false_positive": false_pos,
        "true_negative": true_neg,
        "specificity": float(true_neg / max(1, int((y == 0).sum()))),
        "precision_referable_action": float(decided_pos / max(1, int(decided.sum()))),
        "referable_action_rate": float(decided.mean()),
        "missed_risk_ucb95": ucb,
        "certifies_5_percent_95_confidence": bool(math.isfinite(ucb) and ucb <= 0.05),
    }


def select_threshold(y: np.ndarray, scores: np.ndarray, factors: list[float], min_positive: int) -> dict[str, Any]:
    positives = scores[y == 1]
    best = None
    for factor in factors:
        threshold = max(0.0, float(np.min(positives)) * factor)
        row = eval_threshold(y, scores, threshold)
        row["threshold_name"] = f"positive_floor_x{factor:g}"
        if row["missed_positive"] != 0 or row["decided_positive"] < min_positive:
            continue
        if best is None or (
            row["specificity"],
            row["precision_referable_action"],
            -row["referable_action_rate"],
        ) > (
            best["specificity"],
            best["precision_referable_action"],
            -best["referable_action_rate"],
        ):
            best = row
    return best or {"threshold": 1.01, "threshold_name": "no_valid_zero_miss_threshold"}


def write_predictions(path: Path, split: str, ids: list[str], y: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "referable_dr", "score", "action"])
        if not exists:
            writer.writeheader()
        for image_id, yy, score in zip(ids, y, scores):
            writer.writerow(
                {
                    "split": split,
                    "image_id": image_id,
                    "referable_dr": int(yy),
                    "score": f"{float(score):.8f}",
                    "action": "AUTONOMOUS_REFERABLE" if score >= threshold else "AUTONOMOUS_NONREFERABLE_CANDIDATE",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-validation-positive", type=int, default=59)
    parser.add_argument("--min-specificity", type=float, default=0.70)
    parser.add_argument("--threshold-safety-factors", default="0.99,0.95,0.9,0.8,0.7,0.5,0.25")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if str(device) == "auto":
        device = torch.device("cpu")

    rows = read_manifest(Path(args.manifest))
    raw = np.load(args.feature_cache, allow_pickle=False)
    features = {k: raw[k].astype("float32") for k in raw.files}
    x_train, y_train, _ = load_xy(rows, features, "train")
    x_valid, y_valid, valid_ids = load_xy(rows, features, "valid")
    x_test, y_test, test_ids = load_xy(rows, features, "test")

    model = MLP(x_train.shape[1], args.hidden, args.dropout).to(device)
    pos_weight = float((y_train == 0).sum() / max(1, int((y_train == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True)
    factors = [float(x) for x in args.threshold_safety_factors.split(",") if x.strip()]

    best_state = None
    best_row = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(yb)
            seen += len(yb)
        valid_scores = predict(model, x_valid, args.batch_size, device)
        selected = select_threshold(y_valid, valid_scores, factors, args.min_validation_positive)
        row = {"epoch": epoch, "train_loss": total / max(1, seen), "valid_selected": selected}
        history.append(row)
        if best_row is None or (
            int(selected.get("missed_positive", 10**9)) == 0,
            float(selected.get("specificity", -1)),
            float(selected.get("precision_referable_action", -1)),
            -float(selected.get("referable_action_rate", 1e9)),
        ) > (
            int(best_row["valid_selected"].get("missed_positive", 10**9)) == 0,
            float(best_row["valid_selected"].get("specificity", -1)),
            float(best_row["valid_selected"].get("precision_referable_action", -1)),
            -float(best_row["valid_selected"].get("referable_action_rate", 1e9)),
        ):
            best_row = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or float(selected.get("specificity", 0)) >= args.min_specificity:
            print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_scores = predict(model, x_valid, args.batch_size, device)
    selected = select_threshold(y_valid, valid_scores, factors, args.min_validation_positive)
    threshold = float(selected["threshold"])
    valid_cert = eval_threshold(y_valid, valid_scores, threshold)
    test_scores = predict(model, x_test, args.batch_size, device)
    test_cert = eval_threshold(y_test, test_scores, threshold)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, "valid", valid_ids, y_valid, valid_scores, threshold)
    write_predictions(pred_path, "test", test_ids, y_test, test_scores, threshold)
    artifact = {
        "model_state": model.cpu().state_dict(),
        "input_dim": int(x_train.shape[1]),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "threshold": threshold,
        "best_epoch": best_row,
    }
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.artifact)
    joblib.dump({"threshold": threshold, "best_epoch": best_row}, str(Path(args.artifact).with_suffix(".joblib")))
    summary = {
        "status": (
            "GO_FEATURE_MLP_SINGLE_IMAGE_AUTONOMY"
            if test_cert["certifies_5_percent_95_confidence"] and test_cert["specificity"] >= args.min_specificity
            else "NO_GO_FEATURE_MLP_SINGLE_IMAGE_AUTONOMY"
        ),
        "manifest": args.manifest,
        "feature_cache": args.feature_cache,
        "artifact": args.artifact,
        "best_epoch": best_row,
        "valid_certificate": valid_cert,
        "test_certificate": test_cert,
        "predictions": str(pred_path),
        "history_tail": history[-20:],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
