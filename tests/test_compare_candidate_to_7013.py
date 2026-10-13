import csv
import tempfile
import unittest
from pathlib import Path

from active_retina.compare_candidate_to_7013 import evaluate


class CandidateComparisonTests(unittest.TestCase):
    def test_aggregates_views_and_counts_incremental_clears_without_rewriting_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.csv"
            manifest = root / "manifest.csv"
            predictions = root / "predictions.csv"
            self._write(reference, [
                {"source": "X", "eye_id": "n1", "grade": "0", "referable": "0", "ungradable": "0", "action": "REFER"},
                {"source": "X", "eye_id": "n2", "grade": "1", "referable": "0", "ungradable": "0", "action": "CLEAR"},
                {"source": "X", "eye_id": "p1", "grade": "2", "referable": "1", "ungradable": "0", "action": "REFER"},
                {"source": "X", "eye_id": "u1", "grade": "5", "referable": "0", "ungradable": "1", "action": "REFER"},
            ])
            self._write(manifest, [
                {"image_id": "n1a", "eye_id": "n1", "source_dataset": "X"},
                {"image_id": "n1b", "eye_id": "n1", "source_dataset": "X"},
                {"image_id": "n2a", "eye_id": "n2", "source_dataset": "X"},
                {"image_id": "p1a", "eye_id": "p1", "source_dataset": "X"},
                {"image_id": "u1a", "eye_id": "u1", "source_dataset": "X"},
            ])
            self._write(predictions, [
                {"image_id": "n1a", "score": "-1", "quality_score": "-1"},
                {"image_id": "n1b", "score": "-2", "quality_score": "-2"},
                {"image_id": "n2a", "score": "-3", "quality_score": "-2"},
                {"image_id": "p1a", "score": "1", "quality_score": "-2"},
                {"image_id": "u1a", "score": "-5", "quality_score": "2"},
            ])

            result = evaluate(reference, manifest, predictions)

            self.assertEqual(result["frozen_reference"]["true_negatives"], 1)
            self.assertEqual(result["candidate_standalone"]["missed_referable"], 0)
            self.assertEqual(result["candidate_standalone"]["missed_ungradable"], 0)
            self.assertEqual(result["reference_plus_candidate_clearance"]["true_negatives"], 2)
            self.assertEqual(result["reference_plus_candidate_clearance"]["additional_true_negatives"], 1)
            self.assertEqual(result["reference_plus_candidate_clearance"]["cases_needed_from_frozen_reference"], 1)
            self.assertEqual(result["reference_plus_candidate_clearance"]["additional_true_negatives_needed_from_this_result"], 0)

    @staticmethod
    def _write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
