"""Extract frozen pretrained ResNet features from MMRDR development rows only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_every_image_autonomy import FundusDataset, GradeReferableModel, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="resnet50.a1_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows or any(row["source_split"] not in {"train", "valid"} for row in rows):
        raise SystemExit("Refusing feature extraction: development-only train/valid manifest required.")
    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    wrapper = GradeReferableModel(args.model_name, args.image_size, pretrained=True)
    model = wrapper.model.to(device).eval()
    dataset = FundusDataset(rows, wrapper.eval_transform, Path.cwd())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    features: list[np.ndarray] = []
    grades: list[int] = []
    labels: list[int] = []
    ids: list[str] = []
    with torch.inference_mode():
        for batch_index, (images, batch_grades, batch_labels, batch_ids) in enumerate(loader, 1):
            fmap = model.forward_features(images.to(device))
            pooled = model.forward_head(fmap, pre_logits=True)
            features.append(pooled.float().cpu().numpy())
            grades.extend(int(x) for x in batch_grades.numpy())
            labels.extend(int(x) for x in batch_labels.numpy())
            ids.extend(str(x) for x in batch_ids)
            if batch_index % 50 == 0:
                print(f"batches={batch_index}/{len(loader)} images={len(ids)}/{len(rows)}", flush=True)

    feature_array = np.concatenate(features).astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feature_array,
        grades=np.asarray(grades, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        image_ids=np.asarray(ids, dtype=str),
        splits=np.asarray([row["source_split"] for row in rows], dtype=str),
        source_datasets=np.asarray([row.get("source_dataset", "unknown") for row in rows], dtype=str),
        model_name=np.asarray(args.model_name),
        image_size=np.asarray(args.image_size),
    )
    print(f"cached_images={len(ids)} feature_dim={feature_array.shape[1]} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
