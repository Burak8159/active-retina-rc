"""Fit and select a referable-risk head on cached MMRDR train/validation features."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .train_every_image_autonomy import (
    GradeReferableModel,
    class_weights,
    metrics,
    pairwise_boundary_loss,
    read_manifest,
    referable_logit,
    set_seed,
    select_zero_miss_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--binary-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    set_seed(args.seed)

    rows = read_manifest(args.manifest)
    if not rows or any(row["source_split"] not in {"train", "valid"} for row in rows):
        raise SystemExit("Refusing head training: development-only train/valid manifest required.")
    with np.load(args.features, allow_pickle=False) as cache:
        x = torch.from_numpy(cache["features"].copy())
        y_grade = torch.from_numpy(cache["grades"].copy()).long()
        y_binary = torch.from_numpy(cache["labels"].copy()).long()
        ids = cache["image_ids"].astype(str).tolist()
        splits = cache["splits"].astype(str)
        if cache["model_name"].item() != args.model_name or int(cache["image_size"].item()) != args.image_size:
            raise SystemExit("Feature-cache model or input size does not match training arguments.")
    if ids != [str(row["image_id"]) for row in rows] or len(splits) != len(rows):
        raise SystemExit("Feature cache row order/identity does not match its manifest.")

    train_mask = torch.from_numpy(splits == "train")
    valid_mask = torch.from_numpy(splits == "valid")
    if not train_mask.any() or not valid_mask.any() or torch.any((splits != "train") & (splits != "valid")):
        raise SystemExit("Feature cache must contain only non-empty train and validation partitions.")
    train_rows = [row for row in rows if row["source_split"] == "train"]
    weights = class_weights(train_rows, 1.0, 1.0, 1.0, "inverse")
    head = nn.Linear(x.shape[1], 6)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best = None
    history = []
    stale = 0
    train_indices = torch.where(train_mask)[0]
    valid_indices = torch.where(valid_mask)[0]
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = train_indices[torch.randperm(len(train_indices))]
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = head(x[idx])
            boundary = referable_logit(logits.float())
            loss = criterion(logits, y_grade[idx])
            loss = loss + args.binary_weight * nn.functional.binary_cross_entropy_with_logits(boundary, y_binary[idx].float())
            loss = loss + args.rank_weight * pairwise_boundary_loss(boundary, y_binary[idx], args.rank_margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
            seen += len(idx)

        head.eval()
        with torch.no_grad():
            valid_scores = referable_logit(head(x[valid_indices]).float()).numpy()
        valid_labels = y_binary[valid_indices].numpy()
        selected = select_zero_miss_threshold(valid_labels, valid_scores)
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, seen), "validation_zero_miss": selected}
        history.append(row)
        key = (int(selected["false_negative_clearance"] == 0), float(selected["specificity"]))
        if best is None or key > best[0]:
            best = (key, selected, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, valid_scores.copy())
            stale = 0
        else:
            stale += 1
        (args.output_dir / "progress.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"history": history, "test_evaluated": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    if best is None:
        raise RuntimeError("No cached-head checkpoint selected.")
    _, selected, head_state, _ = best
    wrapper = GradeReferableModel(args.model_name, args.image_size, pretrained=True)
    model_state = wrapper.model.state_dict()
    model_state["fc.weight"] = head_state["weight"]
    model_state["fc.bias"] = head_state["bias"]
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": args.model_name,
        "state_dict": model_state,
        "threshold": float(selected["threshold"]),
        "validation_metrics": selected,
        "best_epoch": int(max(history, key=lambda h: (h["validation_zero_miss"]["false_negative_clearance"] == 0, h["validation_zero_miss"]["specificity"]))["epoch"]),
        "trainable_info": {"mode": "frozen_backbone_cached_feature_linear_head", "trainable_parameters": int(head.weight.numel() + head.bias.numel())},
        "args": {"manifest": str(args.manifest), "image_size": args.image_size},
        "test_evaluated": False,
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
    }, args.artifact)

    labels = y_binary[valid_indices].numpy()
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
        "dataset": "MMRDR-UWF",
        "model_name": args.model_name,
        "best_epoch": int(max(history, key=lambda h: (h["validation_zero_miss"]["false_negative_clearance"] == 0, h["validation_zero_miss"]["specificity"]))["epoch"]),
        "validation_metrics_at_zero_miss_threshold": metrics(labels, best[3], float(selected["threshold"])),
        "trainable_info": {"mode": "frozen_backbone_cached_feature_linear_head", "trainable_parameters": int(head.weight.numel() + head.bias.numel())},
        "checkpoint": str(args.artifact),
        "test_evaluated": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
