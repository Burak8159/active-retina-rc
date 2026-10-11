from __future__ import annotations

import csv
from io import BytesIO
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from active_retina.prepare_idrid_training import build_manifest


class TestPrepareIdridTraining(unittest.TestCase):
    def test_extracts_only_explicit_training_images_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "idrid.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                rows = []
                for index in range(3):
                    name = f"IDRiD_{index + 1:03d}"
                    image_path = root / f"{name}.jpg"
                    Image.new("RGB", (8, 8), (index * 60 + 1, index * 60 + 2, index * 60 + 3)).save(image_path)
                    bundle.write(image_path, f"1. Original Images/a. Training Set/{name}.jpg")
                    rows.append((name, index))
                bundle.write(root / "IDRiD_003.jpg", "1. Original Images/b. Testing Set/IDRiD_003.jpg")
                labels_path = root / "labels.csv"
                with labels_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("Image name", "Retinopathy grade", "Risk of macular edema"))
                    for name, grade in rows:
                        writer.writerow((name, grade, 0))
                bundle.write(labels_path, "2. Groundtruths/IDRiD_Disease Grading_Training Labels.csv")

            manifest = root / "manifest.csv"
            summary_path = root / "summary.json"
            summary = build_manifest(
                archive, root / "images", manifest, summary_path, expected_images=3
            )
            with manifest.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 3)
            self.assertTrue(all(row["source_split"] == "idrid_train" for row in records))
            self.assertEqual([int(row["grade"]) for row in records], [0, 1, 2])
            self.assertEqual(summary["n_training_images"], 3)
            self.assertFalse(summary["test_partition_used"])

    def test_excludes_same_grade_repeats_and_both_sides_of_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "duplicates.zip"
            rows = [("IDRiD_001", 1), ("IDRiD_002", 1), ("IDRiD_003", 2), ("IDRiD_004", 3), ("IDRiD_005", 4)]
            def jpeg(color: tuple[int, int, int]) -> bytes:
                stream = BytesIO()
                Image.new("RGB", (8, 8), color).save(stream, format="JPEG")
                return stream.getvalue()

            contents = {
                "IDRiD_001": jpeg((10, 20, 30)),
                "IDRiD_002": jpeg((10, 20, 30)),
                "IDRiD_003": jpeg((40, 50, 60)),
                "IDRiD_004": jpeg((40, 50, 60)),
                "IDRiD_005": jpeg((70, 80, 90)),
            }
            labels_path = root / "labels.csv"
            with labels_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Image name", "Retinopathy grade"))
                writer.writerows(rows)
            with zipfile.ZipFile(archive, "w") as bundle:
                for name, raw in contents.items():
                    bundle.writestr(f"1. Original Images/a. Training Set/{name}.jpg", raw)
                bundle.write(labels_path, "2. Groundtruths/IDRiD_Disease Grading_Training Labels.csv")

            manifest = root / "manifest.csv"
            summary = build_manifest(archive, root / "images", manifest, root / "summary.json", expected_images=5)
            with manifest.open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([row["image_id"] for row in records], ["idrid_train:IDRiD_001", "idrid_train:IDRiD_005"])
            self.assertEqual(summary["n_exact_duplicate_rows_excluded"], 1)
            self.assertEqual(summary["n_conflicting_duplicate_rows_excluded"], 2)


if __name__ == "__main__":
    unittest.main()
