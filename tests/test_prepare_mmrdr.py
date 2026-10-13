import csv
from pathlib import Path

from PIL import Image

from active_retina.prepare_mmrdr import build_manifest


def test_mmrdr_manifest_keeps_official_test_and_maps_referable_grades(tmp_path: Path):
    root = tmp_path / "MMRDR"
    image_dir = root / "MMRDR-UWF" / "img"
    image_dir.mkdir(parents=True)
    rows = []
    for i in range(30):
        name = f"tr{i + 1:06d}.jpg"
        Image.new("RGB", (8, 8), color=(i, 0, 0)).save(image_dir / name)
        rows.append({"type": "train", "image": name, "grade": str(i % 3), "lesion": "[]", "lr": str(i % 2)})
    for i in range(6):
        name = f"ts{i + 1:06d}.jpg"
        Image.new("RGB", (8, 8), color=(0, i, 0)).save(image_dir / name)
        rows.append({"type": "test", "image": name, "grade": str(i % 3), "lesion": "[]", "lr": str(i % 2)})
    csv_path = root / "MMRDR-UWF" / "UWF.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "image", "grade", "lesion", "lr"])
        writer.writeheader()
        writer.writerows(rows)

    records, summary = build_manifest(root, "UWF", validation_fraction=0.25, seed=4)

    official_test = [row for row in records if row["image_id"].endswith(tuple(f"ts{i:06d}" for i in range(1, 7)))]
    assert len(official_test) == 6
    assert all(row["source_split"] == "test" for row in official_test)
    assert all(row["referable_dr"] == int(row["grade"] >= 2) for row in records)
    assert summary["official_test_patient_level"] is True
    assert summary["patient_ids_available_in_exported_table"] is False
    assert summary["split_counts"]["test"]["images"] == 6
