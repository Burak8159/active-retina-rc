from __future__ import annotations

import unittest

from active_retina.append_training_only_manifest import combine


def row(image_id: str, split: str, source: str, grade: int) -> dict[str, str]:
    return {
        "patient_id": image_id,
        "eye_id": image_id,
        "image_id": image_id,
        "field": "single_fundus",
        "grade": str(grade),
        "referable_dr": str(int(grade >= 2)),
        "image_path": f"external/{image_id}.jpg",
        "source_split": split,
        "source_dataset": source,
    }


class TestAppendTrainingOnlyManifest(unittest.TestCase):
    def test_appends_training_rows_without_mutating_base(self) -> None:
        base = [row("base-valid", "valid", "ddr_hf_specificity300", 1), row("base-test", "test", "ddr_hf_specificity300", 2)]
        snapshot = [dict(item) for item in base]
        added = [row("aptos-1", "aptos_hf_train", "aptos_hf", 1)]
        result = combine(base, added)
        self.assertEqual(result[: len(base)], snapshot)
        self.assertEqual(result[-1], added[0])

    def test_rejects_nontraining_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "training rows only"):
            combine([row("base", "train", "ddr_hf_specificity300", 0)], [row("aptos", "aptos_hf_valid", "aptos_hf", 0)])

    def test_rejects_duplicate_image_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap or duplicate"):
            combine([row("same", "train", "ddr_hf_specificity300", 0)], [row("same", "aptos_hf_train", "aptos_hf", 0)])

    def test_rejects_inconsistent_binary_mapping(self) -> None:
        item = row("aptos", "aptos_hf_train", "aptos_hf", 1)
        item["referable_dr"] = "1"
        with self.assertRaisesRegex(ValueError, "grade or referable"):
            combine([row("base", "train", "ddr_hf_specificity300", 0)], [item])


if __name__ == "__main__":
    unittest.main()
