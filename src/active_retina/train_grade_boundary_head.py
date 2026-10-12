"""Train source-specific grade-boundary heads from cached ResNet50 features.

This is development tooling. Model selection and zero-miss thresholds use
previously inspected validation labels and cannot certify clinical risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "ddr": {
        "manifest": "results/ddr_hf_specificity300_manifest.csv",
        "features": "results/ddr_hf_specificity300_features.npz",
        "train_split": "ddr_train",
        "eval_split": "ddr_valid",
    },
    "deepdrid": {
        "manifest": "results/deepdrid_manifest.csv",
        "features": "results/deepdrid_resnet50_features.npz",
        "train_split": "official_train",
        "eval_split": "official_validation",
    },
    "drtid": {
        "manifest": "results/drtid_untouched_certification/certification_manifest.csv",
        "features": "results/drtid_untouched_certification/certification_features.npz",
        "train_split": "drtid_training",
        # This test partition has already been inspected elsewhere; development only.
        "eval_split": "drtid_testing",
    },
}

BASELINE_INPUTS = {
    "ddr": "results/rtx_aptos_replay_laststage_20261011/validation_predictions.csv",
    "deepdrid": "results/rtx_fundus_b5_all_sources_20261011/validation_predictions.csv",
    "drtid": "results/rtx_warmstart_geometry_corrected_20261011/pooled_predictions_tta_flips.csv",
}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _comparison_eye_ids(root: Path, source: str) -> set[str]:
    rows = _read_manifest(root / BASELINE_INPUTS[source])
    if source == "ddr":
        rows = [row for row in rows if row.get("split") == "valid"
                and row.get("source_dataset") == "ddr_hf_specificity300"
                and row.get("eye_id") != "ddr_hf_specificity300:ddr_test_20170417105537128"]
    elif source == "deepdrid":
        rows = [row for row in rows if row.get("split") == "valid"
                and row.get("source_dataset") == "deepdrid_paired"]
    return {row["eye_id"] for row in rows}


def _eye_rows(rows: list[dict[str, str]], features: Any, split: str | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if split is None or row["source_split"] == split:
            grouped[row["eye_id"]].append(row)

    output: list[dict[str, Any]] = []
    for eye_id, group in sorted(grouped.items()):
        grades = {int(row["grade"]) for row in group}
        if len(grades) != 1:
            raise ValueError(f"inconsistent eye grade: {eye_id}: {grades}")
        vectors = [np.asarray(features[row["image_id"]], dtype=np.float32) for row in group]
        if not vectors or any(vector.shape != vectors[0].shape for vector in vectors):
            raise ValueError(f"invalid/missing feature vector: {eye_id}")
        stack = np.stack(vectors)
        mean = stack.mean(axis=0)
        spread = np.abs(stack[0] - stack[1]) if len(stack) == 2 else np.zeros_like(mean)
        output.append({
            "eye_id": eye_id,
            "grade": grades.pop(),
            "x": np.concatenate((mean, spread, np.asarray([len(stack) == 2], np.float32))),
            "n_views": len(stack),
        })
    return output


class BoundaryMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def _metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    grades = np.asarray([row["grade"] for row in rows])
    positives = grades >= 2
    negatives = grades < 2
    if not positives.any() or not negatives.any():
        raise ValueError("evaluation split must contain both referable and nonreferable grades")
    cutoff = float(scores[positives].min())
    referred = scores >= cutoff
    true_negative = int(np.sum(negatives & ~referred))
    return {
        "n_eyes": int(len(rows)),
        "positive_eyes": int(positives.sum()),
        "negative_eyes": int(negatives.sum()),
        "true_negative": true_negative,
        "false_positive_referral": int(np.sum(negatives & referred)),
        "missed_referable": int(np.sum(positives & ~referred)),
        "specificity_percent": 100.0 * true_negative / int(negatives.sum()),
        "binary_action_coverage_percent": 100.0,
        "posthoc_min_positive_score_cutoff": cutoff,
        "grade1_negative_count": int(np.sum(grades == 1)),
        "grade1_cleared_count": int(np.sum((grades == 1) & ~referred)),
        "grade2_positive_count": int(np.sum(grades == 2)),
        "evaluation_status": "EXPOSED_DEVELOPMENT_ONLY_POSTHOC_THRESHOLD",
    }


def _train_one(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], *,
    epochs: int, seed: int, hidden: int, lr: float, grade1_weight: float,
    pairwise_weight: float, device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    x_train = np.stack([row["x"] for row in train_rows])
    x_eval = np.stack([row["x"] for row in eval_rows])
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True).clip(1e-5)
    x_train = np.clip((x_train - mu) / sigma, -8.0, 8.0)
    x_eval = np.clip((x_eval - mu) / sigma, -8.0, 8.0)
    y_train = np.asarray([row["grade"] >= 2 for row in train_rows], dtype=np.float32)
    grades = np.asarray([row["grade"] for row in train_rows])
    weights = np.where(grades == 1, grade1_weight, 1.0).astype(np.float32)
    weights = np.where(y_train > 0, 2.0, weights).astype(np.float32)

    tx = torch.tensor(x_train, device=device)
    ty = torch.tensor(y_train, device=device)
    tw = torch.tensor(weights, device=device)
    model = BoundaryMLP(tx.shape[1], hidden, 0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    batch_size = min(256, len(train_rows))

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            logits = model(tx[idx])
            bce = (nn.functional.binary_cross_entropy_with_logits(logits, ty[idx], reduction="none") * tw[idx]).mean()
            pos = logits[ty[idx] > 0]
            neg = logits[ty[idx] <= 0]
            if len(pos) and len(neg) and pairwise_weight:
                pairwise = nn.functional.softplus(1.0 - (pos[:, None] - neg[None, :])).mean()
            else:
                pairwise = logits.new_zeros(())
            loss = bce + pairwise_weight * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x_eval, device=device)).cpu().numpy()
    return scores, {"epochs": epochs, "seed": seed, "hidden": hidden, "lr": lr,
                    "grade1_weight": grade1_weight, "pairwise_weight": pairwise_weight}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = Path(args.root)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "DEVELOPMENT_ONLY_NOT_CERTIFICATION",
        "method": "source-specific frozen-ResNet50 feature MLP; symmetric pair mean/absolute-difference; weighted binary referable loss plus pairwise ranking",
        "data_rule": "fit exclusively on each dataset's named training split; use named validation/test splits only for exposed development evaluation",
        "candidates": {},
    }
    prediction_rows: list[dict[str, Any]] = []

    for source in args.sources:
        cfg = SOURCES[source]
        manifest = _read_manifest(root / cfg["manifest"])
        features = np.load(root / cfg["features"], allow_pickle=False)
        eval_rows = _eye_rows(manifest, features, None)
        comparison_ids = _comparison_eye_ids(root, source)
        # The legacy DDR prediction export prefixes eye IDs with the dataset key.
        normalized_ids = {value.split(":", 1)[-1] for value in comparison_ids}
        eval_rows = [row for row in eval_rows if row["eye_id"] in normalized_ids]
        if len(eval_rows) != len(normalized_ids):
            available = {row["eye_id"] for row in eval_rows}
            missing = sorted(normalized_ids - available)
            raise ValueError(f"{source}: common comparison IDs missing from manifest: {missing[:5]}")
        train_rows = _eye_rows(manifest, features, cfg["train_split"])
        overlap = {row["eye_id"] for row in train_rows} & normalized_ids
        train_rows = [row for row in train_rows if row["eye_id"] not in normalized_ids]
        source_trials = []
        for seed in args.seeds:
            for grade1_weight, pairwise_weight in args.losses:
                scores, hyper = _train_one(
                    train_rows, eval_rows, epochs=args.epochs, seed=seed,
                    hidden=args.hidden, lr=args.lr, grade1_weight=grade1_weight,
                    pairwise_weight=pairwise_weight, device=device,
                )
                metrics = _metrics(eval_rows, scores)
                trial = {**hyper, **metrics}
                source_trials.append(trial)
                for row, score in zip(eval_rows, scores):
                    prediction_rows.append({
                        "source": source, "eye_id": row["eye_id"], "grade": row["grade"],
                        "score": float(score), "candidate": f"s{seed}_g{grade1_weight}_r{pairwise_weight}",
                    })
        source_trials.sort(key=lambda item: (item["specificity_percent"], -item["false_positive_referral"]), reverse=True)
        result["candidates"][source] = {
            "n_train_eyes": len(train_rows),
            "comparison_ids_removed_from_training": len(overlap),
            "n_eval_eyes": len(eval_rows),
            "best_validation_candidate": source_trials[0],
            "all_validation_candidates": source_trials,
        }

    pooled_tn = sum(item["best_validation_candidate"]["true_negative"] for item in result["candidates"].values())
    pooled_negative = sum(item["best_validation_candidate"]["negative_eyes"] for item in result["candidates"].values())
    result["pooled_best_by_source"] = {
        "true_negative": pooled_tn,
        "gradable_nonreferable_denominator": pooled_negative,
        "specificity_percent": 100.0 * pooled_tn / pooled_negative,
        "observed_missed_referable": sum(item["best_validation_candidate"]["missed_referable"] for item in result["candidates"].values()),
        "binary_action_coverage_percent": 100.0,
        "comparison_reference": "70.13% = 270/385; same exposed evaluation eyes",
        "status": "POSTHOC_DEVELOPMENT_COMPARISON_NOT_CERTIFICATION",
    }

    with (out / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "eye_id", "grade", "score", "candidate"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", default="results/grade_boundary_head_20261012")
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--losses", type=float, nargs="+", default=[1.0, 0.5, 2.0, 1.0, 4.0, 1.0],
                        help="pairs of grade1_weight, pairwise_weight")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if len(args.losses) % 2:
        parser.error("--losses must contain grade1_weight/pairwise_weight pairs")
    args.losses = list(zip(args.losses[::2], args.losses[1::2]))
    report = run(args)
    for source, item in report["candidates"].items():
        print(json.dumps({"source": source, "train": item["n_train_eyes"], "eval": item["n_eval_eyes"],
                          "best": item["best_validation_candidate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
