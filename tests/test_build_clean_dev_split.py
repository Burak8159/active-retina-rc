import unittest

from active_retina.build_clean_dev_split import build_split


class CleanDevelopmentSplitTests(unittest.TestCase):
    def test_excludes_reference_patient_and_nontraining_partitions(self):
        rows = []
        for i, (grade, split) in enumerate([
            (0, "ddr_train"), (2, "ddr_train"), (0, "ddr_train"), (2, "ddr_train"),
            (0, "ddr_train"), (2, "ddr_train"), (0, "ddr_train"), (2, "ddr_train"),
            (0, "ddr_test"),
        ]):
            patient = "reference-patient" if i == 0 else f"patient-{i}"
            rows.append({
                "source_dataset": "ddr", "patient_id": patient, "eye_id": f"eye-{i}",
                "image_id": f"image-{i}", "grade": str(grade), "original_source_split": split,
                "source_split": "train", "image_path": f"{i}.jpg",
            })
        rows.append({
            "source_dataset": "ddr", "patient_id": "reference-patient", "eye_id": "eye-0-other",
            "image_id": "image-0-other", "grade": "1", "original_source_split": "ddr_train",
            "source_split": "train", "image_path": "other.jpg",
        })
        output, summary = build_split(
            rows, [{"eye_id": "eye-0"}], n_splits=2, fold=0, random_state=17
        )

        self.assertEqual(summary["reference_patient_overlap"], 0)
        self.assertEqual(summary["train_valid_patient_overlap"], 0)
        self.assertNotIn("eye-0", {row["eye_id"] for row in output})
        self.assertNotIn("eye-0-other", {row["eye_id"] for row in output})
        self.assertNotIn("eye-8", {row["eye_id"] for row in output})
        self.assertTrue(any(row["source_split"] == "valid" for row in output))


if __name__ == "__main__":
    unittest.main()
