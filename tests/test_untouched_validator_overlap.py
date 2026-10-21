import csv
import tempfile
import unittest
from pathlib import Path

from active_retina import validate_untouched_cohort as validator


class UntouchedValidatorOverlapTests(unittest.TestCase):
    def test_dataset_aliases_share_namespace(self):
        historical = validator.dataset_namespace(
            {"source_dataset": "DDR-HF-specificity300"}, Path("results/old.csv")
        )
        candidate = validator.dataset_namespace(
            {"source_split": "untouched_external_certification"},
            Path("tmp/cohort.csv"),
            "DDR",
        )
        self.assertEqual(historical, "ddr")
        self.assertEqual(candidate, historical)

    def test_known_records_capture_ids_and_exact_image_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.bin"
            image.write_bytes(b"same-image-content")
            manifest = root / "known.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["patient_id", "eye_id", "image_id", "image_path", "source_dataset"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient_id": "p1",
                        "eye_id": "e1",
                        "image_id": "i1",
                        "image_path": str(image),
                        "source_dataset": "DDR-hf-specificity300",
                    }
                )

            original = validator.KNOWN_DEVELOPMENT_MANIFESTS
            try:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = [manifest]
                identifiers, hashes = validator.known_records()
            finally:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = original

            self.assertIn(("ddr", "p1"), identifiers["patient_ids"])
            self.assertIn(("ddr", "e1"), identifiers["eye_ids"])
            self.assertIn(("ddr", "i1"), identifiers["image_ids"])
            self.assertIn(validator.sha256_file(image), hashes)

    def test_valid_paired_cohort_passes_integrity_and_attestation_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_known = root / "known.csv"
            empty_known.write_text(
                "patient_id,eye_id,image_id,image_path,source_dataset\n", encoding="utf-8"
            )
            manifest = root / "cohort.csv"
            fields = [
                "patient_id", "eye_id", "image_id", "field", "grade",
                "referable_dr", "image_path", "source_split",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(59):
                    for field in ("macula", "optic_disc"):
                        image = root / f"{index}_{field}.bin"
                        image.write_bytes(f"{index}:{field}".encode())
                        writer.writerow(
                            {
                                "patient_id": f"cohort-p{index}",
                                "eye_id": f"cohort-e{index}",
                                "image_id": f"cohort-i{index}-{field}",
                                "field": field,
                                "grade": 2,
                                "referable_dr": 1,
                                "image_path": str(image),
                                "source_split": "untouched_external_certification",
                            }
                        )

            original = validator.KNOWN_DEVELOPMENT_MANIFESTS
            try:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = [empty_known]
                result = validator.validate(
                    manifest,
                    {
                        "untouched_for_active_retina_certification": True,
                        "not_used_for_training_or_threshold_selection": True,
                        "not_inspected_before_freeze": True,
                        "source": "new-cohort",
                    },
                )
            finally:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = original

            self.assertEqual(result["status"], "UNTOUCHED_CERTIFICATION_ELIGIBLE")
            self.assertEqual(result["n_complete_paired_eyes"], 59)

    def test_validator_fails_closed_on_duplicate_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            known = root / "known.csv"
            known.write_text(
                "patient_id,eye_id,image_id,image_path,source_dataset\n", encoding="utf-8"
            )
            image_a = root / "a.bin"
            image_b = root / "b.bin"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            manifest = root / "bad.csv"
            fields = [
                "patient_id", "eye_id", "image_id", "field", "grade",
                "referable_dr", "image_path", "source_split",
            ]
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for field, image in (("macula", image_a), ("optic_disc", image_b)):
                    writer.writerow(
                        {
                            "patient_id": "p1", "eye_id": "e1", "image_id": "duplicate",
                            "field": field, "grade": 2, "referable_dr": 1,
                            "image_path": str(image), "source_split": "untouched_external_certification",
                        }
                    )

            original = validator.KNOWN_DEVELOPMENT_MANIFESTS
            try:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = [known]
                result = validator.validate(manifest, {})
            finally:
                validator.KNOWN_DEVELOPMENT_MANIFESTS = original

            self.assertEqual(result["status"], "NOT_ELIGIBLE_FOR_UNTOUCHED_CERTIFICATION")
            self.assertTrue(any("duplicate image_id" in issue for issue in result["manifest_issue_examples"]))


if __name__ == "__main__":
    unittest.main()
