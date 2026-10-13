import unittest

from active_retina.exposed_expert_policy import score_case


def artifact():
    return {
        "artifact_status": "EXPOSED_REPLAY_ONLY_NOT_CLINICALLY_VALIDATED",
        "sources": {
            "DDR": {
                "risk_threshold": 0.6,
                "quality_threshold": 0.9,
                "references": {
                    "aptos_margin": [0.0, 1.0],
                    "ordinal_attention_margin": [0.0, 1.0],
                    "margin_audit_margin": [0.0, 1.0],
                },
            }
        },
    }


class ExposedExpertPolicyTests(unittest.TestCase):
    def test_fixed_per_case_score_and_quality_route(self):
        policy = artifact()
        features = {
            "aptos_margin": 0.0,
            "ordinal_attention_margin": 0.0,
            "margin_audit_margin": 0.0,
            "quality": 0.0,
        }
        result = score_case(policy, "DDR", features)
        self.assertAlmostEqual(result["risk_score"], 0.5)
        self.assertEqual(result["action"], "CLEAR")
        features["quality"] = 1.0
        self.assertEqual(score_case(policy, "DDR", features)["action"], "REFER")

    def test_missing_quality_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "quality"):
            score_case(artifact(), "DDR", {
                "aptos_margin": 0.0,
                "ordinal_attention_margin": 0.0,
                "margin_audit_margin": 0.0,
            })

    def test_unknown_status_fails_closed(self):
        policy = artifact()
        policy["artifact_status"] = "DEPLOYABLE"
        with self.assertRaisesRegex(ValueError, "Unexpected artifact status"):
            score_case(policy, "DDR", {})


if __name__ == "__main__":
    unittest.main()
