"""Extract frozen RETFound-EyePACS features from MMRDR development rows only."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "vit_large_patch16_224"
CLASS_ORDER = ["0-No DR", "1-Mild NPDR", "2-Moderate NPDR", "3-Severe NPDR", "4-PDR"]


class FundusRows(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform, root: Path) -> None:
        self.rows, self.transform, self.root = rows, transform, root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["grade"]), int(row["referable_dr"]), row["image_id"], row["source_split"]


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--splits", nargs="+", choices=("train", "valid"), default=("train", "valid"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if not all_rows or any(row["source_split"] not in {"train", "valid"} for row in all_rows):
        raise SystemExit("Refusing feature extraction: manifest must contain development rows only.")
    rows = [row for row in all_rows if row["source_split"] in set(args.splits)]
    if not rows:
        raise SystemExit("Requested development split(s) contain no rows.")

    device = torch.device(args.device) if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    import timm
    from torchvision import transforms

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5, img_size=args.image_size, global_pool="avg")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = FundusRows(rows, transform, Path.cwd())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    feature_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    grades: list[int] = []
    labels: list[int] = []
    image_ids: list[str] = []
    splits: list[str] = []
    with torch.inference_mode():
        for batch_index, (images, batch_grades, batch_labels, batch_ids, batch_splits) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                encoded = model.forward_features(images)
                features = model.forward_head(encoded, pre_logits=True)
                logits = model.forward_head(encoded)
            feature_parts.append(features.float().cpu().numpy())
            logit_parts.append(logits.float().cpu().numpy())
            grades.extend(int(value) for value in batch_grades)
            labels.extend(int(value) for value in batch_labels)
            image_ids.extend(str(value) for value in batch_ids)
            splits.extend(str(value) for value in batch_splits)
            if batch_index % 100 == 0:
                print(f"batches={batch_index}/{len(loader)} rows={len(image_ids)}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(feature_parts).astype(np.float32, copy=False),
        logits=np.concatenate(logit_parts).astype(np.float32, copy=False),
        grades=np.asarray(grades, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=str),
        splits=np.asarray(splits, dtype=str),
        source_datasets=np.asarray([row["source_dataset"] for row in rows], dtype=str),
        model_name=np.asarray(MODEL_NAME),
        image_size=np.asarray(args.image_size),
        checkpoint_sha256=np.asarray(checkpoint_sha256(args.checkpoint)),
        class_order=np.asarray(CLASS_ORDER, dtype=str),
    )
    print(
        f"cached_rows={len(image_ids)} feature_dim={feature_parts[0].shape[1]} "
        f"logit_dim={logit_parts[0].shape[1]} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
