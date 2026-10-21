import importlib.util
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).parents[1] / "paper/reproducibility/export_mmrdr_validation_trace.py"
SPEC = importlib.util.spec_from_file_location("export_mmrdr_validation_trace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MmrdrValidationTraceTests(unittest.TestCase):
    def test_tracked_history_selects_epoch_20_by_zero_miss_specificity(self):
        root = Path(__file__).parents[1]
        source = root / "results/mmrdr_uwf_cached_head_20261013/summary.json"
        rows, selected_epoch = MODULE.load_and_validate(source)
        selected = next(row for row in rows if row["epoch"] == selected_epoch)
        self.assertEqual(len(rows), 26)
        self.assertTrue(all(row["false_negative_clearance"] == 0 for row in rows))
        self.assertEqual(selected_epoch, 20)
        self.assertEqual(selected["false_negative_clearance"], 0)
        self.assertEqual(selected["true_negative"], 123)
        self.assertEqual(selected["negative"], 600)
        self.assertEqual(selected["specificity"], 0.205)

    def test_run_card_matches_trace_hash_and_selection(self):
        root = Path(__file__).parents[1]
        trace_path = root / "paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv"
        run_card = json.loads((root / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text())
        trace = run_card["validation_selection_trace"]
        self.assertEqual(trace["sha256"], hashlib.sha256(trace_path.read_bytes()).hexdigest())
        self.assertEqual(trace["recorded_epochs"], 26)
        self.assertEqual(trace["zero_miss_epochs"], 26)
        self.assertEqual(trace["selected_epoch"], 20)
        self.assertEqual(trace["selected_true_negative"], 123)
        self.assertEqual(trace["selected_nonreferable"], 600)

    def test_rejects_nonmaximum_selected_epoch(self):
        root = Path(__file__).parents[1]
        source = root / "results/mmrdr_uwf_cached_head_20261013/summary.json"
        summary = json.loads(source.read_text())
        summary["best_epoch"] = 19
        with TemporaryDirectory() as tmp:
            bad_source = Path(tmp) / "summary.json"
            bad_source.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "not maximum-specificity"):
                MODULE.load_and_validate(bad_source)


if __name__ == "__main__":
    unittest.main()
