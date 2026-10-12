import unittest

import numpy as np

from active_retina.audit_expert_score_oracle_union import rank_fraction, zero_miss_clearance


class ExpertScoreOracleUnionTests(unittest.TestCase):
    def test_rank_fraction_uses_average_rank_for_ties(self):
        np.testing.assert_allclose(rank_fraction(np.array([2.0, 1.0, 1.0, 3.0])), [0.75, 0.375, 0.375, 1.0])

    def test_zero_miss_threshold_routes_positive_and_ungradable_cases(self):
        clear, threshold = zero_miss_clearance(
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0.1, 0.4, 0.5, 0.3]),
        )
        self.assertEqual(threshold, 0.3)
        np.testing.assert_array_equal(clear, [True, False, False, False])

    def test_forced_referral_cases_are_not_cleared(self):
        clear, _ = zero_miss_clearance(
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.1, 0.4, 0.2]),
            np.array([False, True, False]),
        )
        np.testing.assert_array_equal(clear, [True, False, False])


if __name__ == "__main__":
    unittest.main()
