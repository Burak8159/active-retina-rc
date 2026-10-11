from __future__ import annotations

import unittest

import numpy as np

from active_retina.audit_oof_risk_stack import _fit_stack, _fold_metrics, _policy_threshold


class OofRiskStackTests(unittest.TestCase):
    def test_threshold_covers_every_training_positive(self) -> None:
        scores = np.asarray([-1.0, 0.2, 0.7, 1.4])
        grades = np.asarray([0, 1, 2, 3])
        threshold = _policy_threshold(scores, grades)
        self.assertEqual(threshold, 0.7)
        self.assertTrue(np.all(scores[grades >= 2] >= threshold))

    def test_fit_stack_returns_finite_scores_and_selected_columns(self) -> None:
        rng = np.random.default_rng(17)
        grades = np.asarray([0, 1] * 20 + [2, 3] * 20)
        signal = (grades >= 2).astype(float) + rng.normal(0, 0.2, len(grades))
        x = np.column_stack((signal, signal + rng.normal(0, 0.1, len(grades)), rng.normal(size=len(grades))))
        scores, threshold, selected = _fit_stack(x, grades, x[:8], k=2, c=0.1)
        self.assertEqual(len(scores), 8)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(len(selected), 2)

    def test_ungradable_cases_are_fail_closed(self) -> None:
        rows = [
            {"grade": "0", "quality_known": "0", "ungradable": "0"},
            {"grade": "2", "quality_known": "0", "ungradable": "0"},
            {"grade": "5", "quality_known": "1", "ungradable": "1"},
        ]
        metrics = _fold_metrics(rows, np.asarray([False, True, False]))
        self.assertEqual(metrics["missed_ungradable"], 0)
        self.assertEqual(metrics["cleared_negative"], 1)


if __name__ == "__main__":
    unittest.main()
