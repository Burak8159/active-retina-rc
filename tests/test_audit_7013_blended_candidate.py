import json
import tempfile
import unittest
from pathlib import Path

from active_retina.audit_7013_blended_candidate import run


class ExposedBlendReplayTest(unittest.TestCase):
    def test_replay_is_distinct_and_explicitly_posthoc(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run(
                root / "results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv",
                Path(tmp) / "candidate",
            )

            self.assertEqual(summary["baseline"]["true_negatives"], 270)
            self.assertEqual(summary["candidate"]["true_negatives"], 272)
            self.assertEqual(summary["candidate"]["gradable_negatives"], 385)
            self.assertEqual(summary["candidate"]["missed_referable"], 0)
            self.assertEqual(summary["candidate"]["missed_ungradable"], 0)
            self.assertEqual(summary["status"], "POSTHOC_SAME_COHORT_DEVELOPMENT_ONLY_NOT_VALIDATED")
            self.assertIn("same 981-case cohort", summary["provenance_warning"])
            saved = json.loads((Path(tmp) / "candidate/summary.json").read_text())
            self.assertEqual(saved, summary)


if __name__ == "__main__":
    unittest.main()
