"""Fit a referable-risk linear head on cached MMRDR train/validation features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .train_every_image_autonomy import (
    GradeReferableModel,
    class_weights,
    hard_boundary_loss,
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
    parser.add_argument("--ordinal-weight", type=float, default=0.25)
    parser.add_argument("--hard-boundary-weight", type=float, default=1.0)
    parser.add_argument("--positive-margin", type=float, default=2.0)
    parser.add_argument("--negative-margin", type=float, default=1.0)
    parser.add_argument("--hard-positive-weight", type=float, default=10.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-grade2-weight", type=float, default=2.0)
    parser.add_argument("--hard-grade1-weight", type=float, default=1.5)
    parser.add_argument("--hard-top-fraction", type=float, default=0.25)
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
    if not train_mask.any() or not valid_mask.any() or np.any((splits != "train") & (splits != "valid")):
        raise SystemExit("Feature cache must contain only non-empty train and validation partitions.")
    train_rows = [row for row in rows if row["source_split"] == "train"]
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, 1.0, 1.0, 1.0, "inverse"))
    head = nn.Linear(x.shape[1], 6)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_indices = torch.where(train_mask)[0]
    valid_indices = torch.where(valid_mask)[0]
    history: list[dict] = []
    best = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = train_indices[torch.randperm(len(train_indices))]
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = head(x[idx])
            boundary = referable_logit(logits.float())
            loss = args.ordinal_weight * criterion(logits, y_grade[idx])
            loss = loss + args.binary_weight * nn.functional.binary_cross_entropy_with_logits(boundary, y_binary[idx].float())
            loss = loss + args.rank_weight * pairwise_boundary_loss(boundary, y_binary[idx], args.rank_margin)
            loss = loss + args.hard_boundary_weight * hard_boundary_loss(
                boundary,
                y_grade[idx],
                y_binary[idx],
                args.positive_margin,
                args.negative_margin,
                args.hard_positive_weight,
                args.hard_negative_weight,
                args.hard_grade2_weight,
                args.hard_grade1_weight,
                args.hard_top_fraction,
            )
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
        record = {"epoch": epoch, "train_loss": loss_sum / max(1, seen), "validation_zero_miss": selected}
        history.append(record)
        key = (int(selected["false_negative_clearance"] == 0), float(selected["specificity"]))
        if best is None or key > best[0]:
            best = (key, selected, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, valid_scores.copy(), epoch)
            stale = 0
        else:
            stale += 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"history": history, "test_evaluated": False}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    if best is None:
        raise RuntimeError("No cached-head checkpoint selected.")
    _, selected, head_state, best_scores, best_epoch = best
    wrapper = GradeReferableModel(args.model_name, args.image_size, pretrained=True)
    model_state = wrapper.model.state_dict()
    classifier = wrapper.model.get_classifier()
    parameter_names = {id(parameter): name for name, parameter in wrapper.model.named_parameters()}
    weight_key = parameter_names.get(id(classifier.weight))
    bias_key = parameter_names.get(id(classifier.bias)) if classifier.bias is not None else None
    if not weight_key or not bias_key or model_state[weight_key].shape != head_state["weight"].shape:
        raise RuntimeError(f"Classifier head does not match cached linear dimensions for {args.model_name}.")
    model_state[weight_key] = head_state["weight"]
    model_state[bias_key] = head_state["bias"]
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": args.model_name,
        "state_dict": model_state,
        "threshold": float(selected["threshold"]),
        "validation_metrics": selected,
        "best_epoch": best_epoch,
        "trainable_info": {"mode": "frozen_backbone_cached_feature_linear_head", "trainable_parameters": int(head.weight.numel() + head.bias.numel())},
        "args": {"manifest": str(args.manifest), "image_size": args.image_size},
        "test_evaluated": False,
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
    }, args.artifact)
    summary = {
        "status": "DEVELOPMENT_CANDIDATE_TEST_NOT_READ",
        "dataset": "MMRDR-UWF",
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "validation_metrics_at_zero_miss_threshold": metrics(y_binary[valid_indices].numpy(), best_scores, float(selected["threshold"])),
        "trainable_info": {"mode": "frozen_backbone_cached_feature_linear_head", "trainable_parameters": int(head.weight.numel() + head.bias.numel())},
        "checkpoint": str(args.artifact),
        "test_evaluated": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
