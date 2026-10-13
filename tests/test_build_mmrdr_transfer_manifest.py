import csv
from pathlib import Path

import pytest

from active_retina.build_mmrdr_transfer_manifest import build_manifest


def test_cfp_is_independent_training_source_and_test_rows_are_rejected(tmp_path: Path):
    cfp_dir = tmp_path / "cfp"
    cfp_dir.mkdir()
    (cfp_dir / "tr000001.jpg").write_bytes(b"train")
    uwf_path = tmp_path / "uwf.csv"
    with uwf_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "patient_id", "eye_id", "image_id", "field", "grade", "referable_dr",
            "image_path", "source_split", "source_dataset",
        ])
        writer.writeheader()
        writer.writerow({
            "patient_id": "", "eye_id": "", "image_id": "mmrdr_uwf_tr000001",
            "field": "uwf", "grade": 1, "referable_dr": 0,
            "image_path": "external/MMRDR/MMRDR-UWF/img/tr000001.jpg",
            "source_split": "valid", "source_dataset": "mmrdr_uwf",
        })
    cfp_path = tmp_path / "cfp.csv"
    with cfp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "image", "grade"])
        writer.writeheader()
        writer.writerow({"type": "FP", "image": "img/tr000001.jpg", "grade": 2})

    rows, summary = build_manifest(uwf_path, cfp_path, cfp_dir)

    cfp_row = next(row for row in rows if row["source_dataset"] == "mmrdr_cfp")
    assert cfp_row["source_split"] == "train"
    assert cfp_row["image_id"] == "mmrdr_cfp_tr000001"
    assert cfp_row["referable_dr"] == 1
    assert cfp_row["patient_id"] == cfp_row["eye_id"] == ""
    assert "not paired-eye identifiers" in summary["pairing"]

    with cfp_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "image", "grade"])
        writer.writerow({"type": "test", "image": "img/ts000001.jpg", "grade": 4})
    with pytest.raises(ValueError, match="Non-training CFP row"):
        build_manifest(uwf_path, cfp_path, cfp_dir)
