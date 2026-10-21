from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from active_retina.train_scoped_retina_heads import (
    RetinaRows,
    ScopedRetinaNet,
    build_rows,
    deduplicate_exact_images,
    infer,
    read_csv,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot evaluation of a frozen scoped retinal checkpoint.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hygd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = build_rows(root, args.manifest, Path("unused"), Path("unused"), args.hygd,
                      args.seed, include_mmrdr=False)
    rows, duplicate_audit = deduplicate_exact_images(rows)
    saved_split = read_csv(root / args.outdir / "split_manifest.csv")
    if [(r["image_sha256"], r["split"]) for r in rows] != [
        (r["image_sha256"], r["split"]) for r in saved_split
    ]:
        raise RuntimeError("Rebuilt rows do not match the saved split manifest; refusing test evaluation.")
    test_rows = [row for row in rows if row["split"] == "test"]
    counts = Counter(row["source"] for row in test_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final evaluation requires the L4 CUDA device")
    checkpoint = torch.load(root / args.checkpoint, map_location=device, weights_only=False)
    model = ScopedRetinaNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(RetinaRows(test_rows, args.size, False), shuffle=False,
                        batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    metrics = summarize(test_rows, infer(model, loader, device))
    report = {
        "status": "SCOPED_DEVELOPMENT_ONLY_NOT_CLINICAL_VALIDATION",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch_selected_by_validation": checkpoint.get("epoch"),
        "test_split_used_once_after_selection": True,
        "n_test_images": len(test_rows),
        "test_sources": dict(counts),
        "duplicate_audit": duplicate_audit,
        "metrics": metrics,
        "adaptive_task_log_variance": checkpoint.get("task_log_variance"),
        "global_any_disease_binary": "NOT_TRAINED_INVALID_WITH_PARTIAL_SOURCE_LABELS",
    }
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "one_shot_test_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
