import unittest

import numpy as np

from active_retina.evaluate_crossfit_expert_selector import metrics, rank_train_test, split_folds


class ExpertScoreSelectorTests(unittest.TestCase):
    def test_patient_groups_stay_in_one_fold(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
        folds = split_folds(labels, groups, 2, 42)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)

    def test_rank_transform_uses_training_distribution(self):
        train = np.array([[0.0], [1.0], [2.0]])
        test = np.array([[0.5], [2.5]])
        train_rank, test_rank = rank_train_test(train, test)
        self.assertTrue(np.all(np.diff(train_rank[:, 0]) > 0))
        self.assertLess(test_rank[0, 0], test_rank[1, 0])

    def test_forced_referrals_count_as_negative_referrals(self):
        labels = np.array([1, 0, 0])
        forced = np.array([False, True, False])
        scores = np.array([0.9, 0.1, 0.2])
        result = metrics(labels, forced, scores, np.array([0.5, 0.5, 0.5]))
        self.assertEqual(result["misses"], 0)
        self.assertEqual(result["negative"], 2)
        self.assertEqual(result["true_negatives"], 1)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["binary_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
