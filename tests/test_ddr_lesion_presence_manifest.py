import csv

import numpy as np
from PIL import Image

from active_retina.build_ddr_lesion_presence_manifest import build_rows


def test_train_mask_join_excludes_images_assigned_to_heldout_splits(tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "lesion_segmentation" / "images" / "train"
    image_dir.mkdir(parents=True)
    for name, value in (("train.jpg", 1), ("heldout.jpg", 0)):
        Image.new("RGB", (8, 8)).save(image_dir / name)
    for lesion in ("EX", "HE", "MA", "SE"):
        target_dir = dataset / "lesion_segmentation" / "annotations" / "train" / lesion
        target_dir.mkdir(parents=True)
        for name in ("train", "heldout"):
            positive = name == "train" and lesion == "MA"
            Image.fromarray(np.full((8, 8), 255 if positive else 0, dtype=np.uint8)).save(
                target_dir / f"{name}.tif"
            )
    (dataset / "DR_grading").mkdir()
    (dataset / "DR_grading" / "train.txt").write_text("train.jpg 1\nheldout.jpg 2\n", encoding="utf-8")

    manifest = tmp_path / "manifest.csv"
    fields = ("patient_id", "eye_id", "image_id", "field", "grade", "referable_dr", "image_path",
              "source_split", "original_source_split", "source_dataset")
    rows = [
        {"patient_id": "train", "eye_id": "train", "image_id": "train", "field": "single_fundus",
         "grade": "1", "referable_dr": "0", "image_path": "external/DDR/train/train.jpg",
         "source_split": "ddr_hf_specificity300_train", "original_source_split": "ddr_train",
         "source_dataset": "ddr_hf_specificity300"},
        {"patient_id": "heldout", "eye_id": "heldout", "image_id": "heldout", "field": "single_fundus",
         "grade": "2", "referable_dr": "1", "image_path": "external/DDR/valid/heldout.jpg",
         "source_split": "ddr_hf_specificity300_valid", "original_source_split": "ddr_valid",
         "source_dataset": "ddr_hf_specificity300"},
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels, expanded, summary = build_rows(manifest, dataset)

    assert [row["image_basename"] for row in labels] == ["train.jpg"]
    assert labels[0]["ma"] == 1
    assert len(expanded) == 2
    assert summary["heldout_annotation_overlap_excluded"] == 1
    assert summary["new_train_images_added_from_official_split"] == 0
