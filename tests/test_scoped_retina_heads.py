from pathlib import Path
import csv

from PIL import Image

from active_retina.train_scoped_retina_heads import (
    BoundedTaskWeights, build_rows, deduplicate_exact_images, summarize,
)


def _row(path: Path, source: str, split: str, grade: int = -1) -> dict:
    return {
        "image_path": str(path), "source": source, "split": split,
        "patient_id": f"{source}:p", "patient_group_verified": source == "hygd_v1_1_0",
        "image_id": path.name, "dr_grade": grade, "ungradable": -1,
        "lesions": [-1] * 7, "gon": -1,
    }


def test_exact_duplicate_across_splits_keeps_training_copy(tmp_path):
    image_path = tmp_path / "fundus.jpg"
    Image.new("RGB", (12, 12), (100, 80, 60)).save(image_path)
    rows = [_row(image_path, "dataset", "train", 0), _row(image_path, "dataset", "test", 0)]

    kept, audit = deduplicate_exact_images(rows)

    assert len(kept) == 1
    assert kept[0]["split"] == "train"
    assert kept[0]["image_sha256"]
    assert audit["cross_split_duplicate_groups"] == 1
    assert audit["rows_removed"] == 1


def test_conflicting_exact_duplicate_labels_are_quarantined(tmp_path):
    image_path = tmp_path / "fundus.jpg"
    Image.new("RGB", (12, 12), (100, 80, 60)).save(image_path)
    rows = [_row(image_path, "dataset_a", "train", 0), _row(image_path, "dataset_b", "valid", 3)]

    kept, audit = deduplicate_exact_images(rows)

    assert kept == []
    assert audit["conflicting_label_groups_quarantined"] == 1


def test_metrics_are_source_scoped_and_report_patient_id_provenance():
    rows = [
        {"source": "hygd_v1_1_0", "patient_id": "p1", "patient_group_verified": True,
         "gon": 0, "ungradable": -1, "dr_grade": -1, "lesions": [-1] * 7},
        {"source": "hygd_v1_1_0", "patient_id": "p2", "patient_group_verified": True,
         "gon": 1, "ungradable": -1, "dr_grade": -1, "lesions": [-1] * 7},
        {"source": "ddr", "patient_id": "pseudo", "patient_group_verified": False,
         "gon": -1, "ungradable": -1, "dr_grade": 0, "lesions": [-1] * 7},
    ]
    import numpy as np
    metrics = summarize(rows, (np.zeros((3, 5)), np.zeros(3), np.zeros((3, 7)), np.array([-2.0, 2.0, 0.0])))

    assert "glaucoma" in metrics["hygd_v1_1_0"]
    assert metrics["hygd_v1_1_0"]["glaucoma"]["patient_ids_verified"] is True
    assert "glaucoma" not in metrics["ddr"]
    assert metrics["ddr"]["dr_grade_0_to_4"]["patient_ids_verified"] is False
    assert metrics["ddr"]["referable_dr_binary_grade_ge_2"]["n_samples"] == 1


def test_build_rows_can_skip_unstaged_mmrdr(tmp_path):
    image_dir = tmp_path / "external" / "HYGD" / "Images"
    image_dir.mkdir(parents=True)
    image = image_dir / "p1_0.jpg"
    Image.new("RGB", (12, 12), (90, 80, 70)).save(image)
    labels = tmp_path / "external" / "HYGD" / "Labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Patient", "Image Name", "Label"])
        writer.writeheader()
        writer.writerow({"Patient": "p1", "Image Name": image.name, "Label": "GON-"})
    manifest = tmp_path / "empty_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grade", "source_split", "source_dataset", "image_path", "image_id"])
        writer.writeheader()

    rows = build_rows(
        tmp_path, Path("empty_manifest.csv"), Path("missing_mmrdr.csv"),
        Path("missing_labels.csv"), Path("external/HYGD"), seed=7,
        include_mmrdr=False,
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "hygd_v1_1_0"
    assert rows[0]["gon"] == 0


def test_adaptive_task_weights_are_trainable_and_bounded():
    import torch

    weights = BoundedTaskWeights()
    losses = {"dr": torch.tensor(2.0, requires_grad=True),
              "gon": torch.tensor(0.2, requires_grad=True)}
    loss = weights.combine(losses)
    loss.backward()
    assert weights.log_variance["dr"].grad is not None
    with torch.no_grad():
        weights.log_variance["dr"].fill_(10)
    weights.clamp_()
    assert 0.5 <= weights.weights()["dr"] <= 2.0
