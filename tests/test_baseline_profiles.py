import unittest

from active_retina.baseline_profiles import PROFILES, list_profiles, score_profile_case


class BaselineProfileTests(unittest.TestCase):
    def test_both_baselines_are_listed_with_distinct_status(self):
        profiles = {item["profile_id"]: item for item in list_profiles()}
        self.assertEqual(set(profiles), {"70.13-replay", "63.12-posthoc"})
        self.assertEqual(profiles["70.13-replay"]["specificity_percent"], 70.12987012987013)
        self.assertEqual(profiles["63.12-posthoc"]["specificity_percent"], 63.116883116883116)
        self.assertNotEqual(profiles["70.13-replay"]["status"], profiles["63.12-posthoc"]["status"])

    def test_posthoc_profile_fails_closed_for_new_case_inference(self):
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            score_profile_case("63.12-posthoc", "DDR", {})

    def test_7013_requires_policy_artifact(self):
        with self.assertRaisesRegex(ValueError, "explicit policy artifact"):
            score_profile_case("70.13-replay", "DDR", {})

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown baseline profile"):
            score_profile_case("70.13", "DDR", {}, {})


if __name__ == "__main__":
    unittest.main()
