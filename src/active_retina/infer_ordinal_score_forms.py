"""Export alternative ordinal score forms for a saved six-grade checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .train_every_image_autonomy import FundusDataset, GradeReferableModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["source_split"] == "valid"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name")
    image_size = int(checkpoint.get("args", {}).get("image_size", 224))
    if not model_name or "state_dict" not in checkpoint:
        raise SystemExit("Checkpoint lacks model identity or state_dict")

    device = torch.device(args.device)
    wrapper = GradeReferableModel(model_name, image_size, pretrained=False)
    wrapper.model.load_state_dict(checkpoint["state_dict"], strict=True)
    wrapper.model.to(device).eval()
    dataset = FundusDataset(rows, wrapper.eval_transform, Path.cwd())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "eye_id", "patient_id", "source_dataset", "grade", "referable_dr",
               "score_referable_logodds", "score_maxclass_margin", "score_expected_grade",
               "score_grade2_logit", "score_grade2_vs_grade1", "score_referable_probability",
               "score_ungradable_margin"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        with torch.inference_mode():
            offset = 0
            for images, grades, labels, image_ids in loader:
                logits = wrapper.model(images.to(device)).float()
                neg = torch.logsumexp(logits[:, :2], dim=1)
                pos = torch.logsumexp(logits[:, 2:5], dim=1)
                boundary = logits[:, 2] - logits[:, 1]
                expected = (logits.softmax(dim=1) * torch.arange(6, device=device)).sum(dim=1)
                max_margin = logits[:, 2:5].max(dim=1).values - logits[:, :2].max(dim=1).values
                prob = logits.softmax(dim=1)[:, 2:5].sum(dim=1)
                quality_margin = logits[:, 5] - torch.logsumexp(logits[:, :5], dim=1)
                values = zip(
                    image_ids,
                    grades.tolist(),
                    labels.tolist(),
                    (pos - neg).cpu().tolist(),
                    max_margin.cpu().tolist(),
                    expected.cpu().tolist(),
                    logits[:, 2].cpu().tolist(),
                    boundary.cpu().tolist(),
                    prob.cpu().tolist(),
                    quality_margin.cpu().tolist(),
                )
                for image_id, grade, label, odds, margin, expected_grade, g2, g2vsg1, probability, quality_margin in values:
                    row = rows[offset]
                    if image_id != row["image_id"]:
                        raise SystemExit("Inference order does not match manifest IDs")
                    writer.writerow({
                        "image_id": image_id,
                        "eye_id": row["eye_id"],
                        "patient_id": row["patient_id"],
                        "source_dataset": row["source_dataset"],
                        "grade": int(grade),
                        "referable_dr": int(label),
                        "score_referable_logodds": odds,
                        "score_maxclass_margin": margin,
                        "score_expected_grade": expected_grade,
                        "score_grade2_logit": g2,
                        "score_grade2_vs_grade1": g2vsg1,
                        "score_referable_probability": probability,
                        "score_ungradable_margin": quality_margin,
                    })
                    offset += 1
    print(json.dumps({"status": "ORDINAL_SCORE_EXPORT_COMPLETE", "checkpoint": str(args.checkpoint),
                      "n_images": len(rows), "n_cases": len({r['eye_id'] for r in rows}),
                      "checkpoint_best_epoch": checkpoint.get("best_epoch"), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
