import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from active_retina.prepare_mbrset_for_untouched_review import main


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_importer(labels: Path, image_root: Path, manifest: Path, summary: Path) -> dict:
    argv = [
        "prepare_mbrset_for_untouched_review",
        "--labels",
        str(labels),
        "--image-root",
        str(image_root),
        "--output-manifest",
        str(manifest),
        "--output-summary",
        str(summary),
    ]
    with patch.object(sys, "argv", argv):
        main()
    return json.loads(summary.read_text(encoding="utf-8"))


class TestMbrsetImporter(unittest.TestCase):
    def test_fails_closed_without_view_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.csv"
            write_rows(labels, [{"patient": "p001", "file": "img001", "final_icdr": 0, "laterality": "R"}])

            summary = run_importer(labels, root / "images", root / "manifest.csv", root / "summary.json")

            self.assertEqual(summary["status"], "MBRSET_PAIRED_MANIFEST_NOT_CREATED")
            self.assertEqual(summary["missing_required_for_paired_manifest"], ["view_position"])

    def test_writes_canonical_pairs_when_view_position_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.csv"
            write_rows(
                labels,
                [
                    {"patient": "p001", "file": "img001", "final_icdr": 0, "laterality": "R", "view_position": "macula"},
                    {
                        "patient": "p001",
                        "file": "img002",
                        "final_icdr": 0,
                        "laterality": "R",
                        "view_position": "optic_disc",
                    },
                    {"patient": "p002", "file": "img003", "final_icdr": 2, "laterality": "L", "view_position": "macula"},
                    {
                        "patient": "p002",
                        "file": "img004",
                        "final_icdr": 2,
                        "laterality": "L",
                        "view_position": "optic_disc",
                    },
                    {"patient": "p003", "file": "img005", "final_icdr": 1, "laterality": "R", "view_position": "macula"},
                ],
            )
            manifest = root / "manifest.csv"

            summary = run_importer(labels, root / "images", manifest, root / "summary.json")
            with manifest.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(summary["status"], "MBRSET_PAIRED_MANIFEST_CREATED")
            self.assertEqual(summary["paired_eyes"], 2)
            self.assertEqual(summary["positive_paired_patients"], 1)
            self.assertEqual({row["field"] for row in rows}, {"macula", "optic_disc"})
            self.assertEqual({row["referable_dr"] for row in rows}, {"0", "1"})


if __name__ == "__main__":
    unittest.main()
