import tempfile
import unittest
from pathlib import Path

from active_retina.audit_exact_image_duplicates import duplicate_groups, remove_duplicate_rows


class ExactDuplicateAuditTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_file(self, name: str, content: bytes) -> str:
        path = self.root / name
        path.write_bytes(content)
        return str(path)

    def test_exact_duplicate_keeps_training_copy(self):
        train_path = self.make_file("train.jpg", b"same bytes")
        valid_path = self.make_file("valid.jpg", b"same bytes")
        rows = [
            {"image_id": "v", "eye_id": "v", "patient_id": "v", "source_split": "source_valid", "source_dataset": "source", "grade": "0", "image_path": valid_path},
            {"image_id": "t", "eye_id": "t", "patient_id": "t", "source_split": "source_train", "source_dataset": "source", "grade": "0", "image_path": train_path},
        ]

        groups = duplicate_groups(rows, self.root)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["kept_image_id"], "t")
        self.assertEqual(groups[0]["removed_image_ids"], ["v"])
        self.assertEqual([row["image_id"] for row in remove_duplicate_rows(rows, groups)], ["t"])

    def test_distinct_files_are_not_collapsed(self):
        rows = [
            {"image_id": "a", "eye_id": "a", "patient_id": "a", "source_split": "source_train", "source_dataset": "source", "grade": "0", "image_path": self.make_file("a.jpg", b"a")},
            {"image_id": "b", "eye_id": "b", "patient_id": "b", "source_split": "source_valid", "source_dataset": "source", "grade": "0", "image_path": self.make_file("b.jpg", b"b")},
        ]

        self.assertEqual(duplicate_groups(rows, self.root), [])

    def test_cross_source_duplicate_fails_closed(self):
        first = self.make_file("first.jpg", b"same")
        second = self.make_file("second.jpg", b"same")
        rows = [
            {"image_id": "a", "eye_id": "a", "patient_id": "a", "source_split": "one_train", "source_dataset": "one", "grade": "0", "image_path": first},
            {"image_id": "b", "eye_id": "b", "patient_id": "b", "source_split": "two_valid", "source_dataset": "two", "grade": "0", "image_path": second},
        ]

        with self.assertRaisesRegex(ValueError, "multiple dataset sources"):
            duplicate_groups(rows, self.root)

    def test_conflicting_duplicate_grades_fail_closed(self):
        first = self.make_file("first.jpg", b"same")
        second = self.make_file("second.jpg", b"same")
        rows = [
            {"image_id": "a", "eye_id": "a", "patient_id": "a", "source_split": "one_train", "source_dataset": "one", "grade": "0", "image_path": first},
            {"image_id": "b", "eye_id": "b", "patient_id": "b", "source_split": "one_valid", "source_dataset": "one", "grade": "2", "image_path": second},
        ]

        with self.assertRaisesRegex(ValueError, "conflicting grades"):
            duplicate_groups(rows, self.root)


if __name__ == "__main__":
    unittest.main()
