import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "paper/audit_submission_evidence.py"
SPEC = importlib.util.spec_from_file_location("audit_submission_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SubmissionEvidenceAuditTests(unittest.TestCase):
    def test_tracked_mmrdr_aggregates_reconcile(self):
        report = MODULE.audit_mmrdr_aggregate(ROOT)
        self.assertEqual(report["status"], "PASS_MMRDR_AGGREGATE_AND_TRACE_INTEGRITY_NOT_INFERENCE_REPRODUCTION")
        self.assertEqual(report["validation"]["epochs"], 26)
        self.assertEqual(report["validation"]["selected_epoch"], 20)
        self.assertEqual(report["official_test"]["false_negative"], 5)
        self.assertEqual(report["official_test"]["rows"], 2597)
        self.assertEqual(
            [round(value * 100, 2) for value in report["official_test"]["sensitivity_wilson_95ci"]],
            [99.09, 99.83],
        )
        self.assertFalse(report["official_test"]["clinical_certification"])

    def test_tampered_trace_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                MODULE.MMRDR_RUN_CARD,
                Path("results/mmrdr_uwf_cached_head_20261013/summary.json"),
                Path("paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv"),
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            trace = root / "paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv"
            trace.write_text(trace.read_text().replace(",123,", ",122,", 1))
            with self.assertRaisesRegex(AssertionError, "trace hash"):
                MODULE.audit_mmrdr_aggregate(root)

    def test_inconsistent_official_test_metric_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                MODULE.MMRDR_RUN_CARD,
                Path("results/mmrdr_uwf_cached_head_20261013/summary.json"),
                Path("paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv"),
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            card_path = root / MODULE.MMRDR_RUN_CARD
            card = json.loads(card_path.read_text())
            card["official_test_result"]["specificity"] += 0.01
            card_path.write_text(json.dumps(card))
            with self.assertRaisesRegex(AssertionError, "specificity does not reconcile"):
                MODULE.audit_mmrdr_aggregate(root)

    def test_inconsistent_validation_metric_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                MODULE.MMRDR_RUN_CARD,
                Path("results/mmrdr_uwf_cached_head_20261013/summary.json"),
                Path("paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv"),
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            card_path = root / MODULE.MMRDR_RUN_CARD
            card = json.loads(card_path.read_text())
            card["validation_result"]["specificity"] += 0.01
            card_path.write_text(json.dumps(card))
            with self.assertRaisesRegex(AssertionError, "selected validation specificity does not reconcile"):
                MODULE.audit_mmrdr_aggregate(root)


if __name__ == "__main__":
    unittest.main()
