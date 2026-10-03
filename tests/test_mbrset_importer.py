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


def run_importer(
    labels: Path,
    image_root: Path,
    manifest: Path,
    summary: Path,
    suffix_view_map: str | None = None,
) -> dict:
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
    if suffix_view_map:
        argv.extend(["--suffix-view-map", suffix_view_map])
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
            self.assertEqual(summary["missing_required_for_paired_manifest"], ["view_position_or_suffix_view_map"])

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

    def test_suffix_view_map_unlocks_public_mbrset_metadata_after_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels = root / "labels.csv"
            write_rows(
                labels,
                [
                    {"patient": "p001", "file": "p001.1.jpg", "final_icdr": 0, "laterality": "right"},
                    {"patient": "p001", "file": "p001.2.jpg", "final_icdr": 0, "laterality": "right"},
                    {"patient": "p001", "file": "p001.3.jpg", "final_icdr": 2, "laterality": "left"},
                    {"patient": "p001", "file": "p001.4.jpg", "final_icdr": 2, "laterality": "left"},
                ],
            )
            manifest = root / "manifest.csv"

            summary = run_importer(
                labels,
                root / "images",
                manifest,
                root / "summary.json",
                "1:macula,2:optic_disc,3:macula,4:optic_disc",
            )
            with manifest.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(summary["status"], "MBRSET_PAIRED_MANIFEST_CREATED")
            self.assertTrue(summary["suffix_view_map_requires_audit"])
            self.assertEqual(summary["paired_eyes"], 2)
            self.assertEqual(summary["positive_paired_patients"], 1)
            self.assertEqual([row["field"] for row in rows], ["macula", "optic_disc", "macula", "optic_disc"])


if __name__ == "__main__":
    unittest.main()
