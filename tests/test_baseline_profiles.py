import unittest
import json
import csv

from active_retina.baseline_profiles import (
    DEFAULT_6312_REPORT,
    DEFAULT_7013_POLICY,
    PROFILES,
    diagnostic_report_6312,
    dispatch_request,
    list_profiles,
    score_profile_case,
)


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

    def test_bundled_70_policy_replays_component_scores(self):
        policy = json.loads(DEFAULT_7013_POLICY.read_text(encoding="utf-8"))
        result = score_profile_case("70.13-replay", "DDR", {
            "aptos_margin": 0.0,
            "ordinal_attention_margin": 0.0,
            "margin_audit_margin": 0.0,
            "quality": 0.0,
        }, policy)
        self.assertEqual(result["profile_id"], "70.13-replay")
        self.assertIn(result["action"], {"CLEAR", "REFER"})

    def test_bundled_6312_is_diagnostic_not_inference(self):
        report = diagnostic_report_6312()
        point = report["best_zero_miss_point_on_inspected_folds"]
        self.assertEqual(report["status"], PROFILES["63.12-posthoc"]["status"])
        self.assertEqual(point["misses"], 0)
        self.assertAlmostEqual(point["specificity_percent"], 63.116883116883116)

    def test_7013_bundled_policy_reproduces_981_case_replay(self):
        policy = json.loads(DEFAULT_7013_POLICY.read_text(encoding="utf-8"))
        replay = DEFAULT_7013_POLICY.parents[2] / "results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv"
        true_negatives = misses = ungradable_misses = 0
        negatives = 0
        with replay.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source = row["source"]
                names = {
                    "DDR": ("aptos_margin", "ordinal_attention_margin", "margin_audit_margin", "quality"),
                    "DeepDRiD": ("ordinal_quality", "sourcebalanced_margin", "fundus_risk"),
                    "DRTiD": ("rankheavy_pair", "fundus_margin", "geometry_pair"),
                }[source]
                features = {name: float(row[name]) for name in names}
                action = score_profile_case("70.13-replay", source, features, policy)["action"]
                positive = row["referable"] == "1"
                ungradable = row["ungradable"] == "1"
                if not positive and not ungradable:
                    negatives += 1
                    true_negatives += action == "CLEAR"
                misses += positive and action != "REFER"
                ungradable_misses += ungradable and action != "REFER"
        self.assertEqual((true_negatives, negatives), (270, 385))
        self.assertEqual((misses, ungradable_misses), (0, 0))

    def test_local_api_serves_both_profiles_and_rejects_6312_scoring(self):
        policy = json.loads(DEFAULT_7013_POLICY.read_text(encoding="utf-8"))
        status, profiles = dispatch_request("GET", "/profiles", {}, policy)
        self.assertEqual(status, 200)
        self.assertEqual(len(profiles), 2)
        status, report = dispatch_request("GET", "/diagnostics/63.12-posthoc", {}, policy)
        self.assertEqual(status, 200)
        self.assertEqual(report["best_zero_miss_point_on_inspected_folds"]["misses"], 0)
        status, error = dispatch_request("POST", "/score", {
            "profile_id": "63.12-posthoc", "source": "DDR", "features": {},
        }, policy)
        self.assertEqual(status, 422)
        self.assertIn("evaluation-only", error["error"])
        status, prediction = dispatch_request("POST", "/score", {
            "profile_id": "70.13-replay", "source": "DDR", "features": {
                "aptos_margin": 0.0,
                "ordinal_attention_margin": 0.0,
                "margin_audit_margin": 0.0,
                "quality": 0.0,
            },
        }, policy)
        self.assertEqual(status, 200)
        self.assertEqual(prediction["profile_id"], "70.13-replay")


if __name__ == "__main__":
    unittest.main()
