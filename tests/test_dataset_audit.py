"""Focused tests for the read-only dataset audit."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.audit import DatasetNotFoundError, audit_dataset, write_audit_artifacts


class DatasetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "chest_xray"
        for split in ("train", "val", "test"):
            for class_name in ("NORMAL", "PNEUMONIA"):
                (self.root / split / class_name).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_image(
        self,
        relative_path: str,
        value: int = 0,
        size: tuple[int, int] = (12, 8),
        mode: str = "L",
    ) -> Path:
        path = self.root / relative_path
        Image.new(mode, size, color=value).save(path)
        return path

    def test_split_class_counts_and_metadata(self) -> None:
        self._write_image("train/NORMAL/normal.png", value=10, size=(12, 8))
        self._write_image(
            "train/PNEUMONIA/pneumonia.png", value=20, size=(9, 7), mode="RGB"
        )
        self._write_image("val/NORMAL/validation.png", value=30)
        self._write_image("test/PNEUMONIA/test.png", value=40)

        result = audit_dataset(self.root)

        self.assertEqual(result.total_images, 4)
        self.assertEqual(result.split_counts(), {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(result.class_counts(), {"NORMAL": 2, "PNEUMONIA": 2})
        grayscale = next(r for r in result.records if r.relative_path.endswith("normal.png"))
        self.assertEqual((grayscale.width, grayscale.height), (12, 8))
        self.assertEqual(grayscale.channel_count, 1)
        self.assertEqual(grayscale.image_mode, "GRAYSCALE")
        self.assertTrue(grayscale.readable)
        self.assertEqual(len(grayscale.sha256 or ""), 64)

    def test_within_split_duplicate_detection(self) -> None:
        original = self._write_image("train/NORMAL/first.png", value=50)
        shutil.copyfile(original, self.root / "train/NORMAL/second.png")

        result = audit_dataset(self.root)

        self.assertEqual(len(result.within_split_duplicates), 1)
        self.assertFalse(result.within_split_duplicates[0].is_cross_split)

    def test_cross_split_and_conflicting_label_duplicate_detection(self) -> None:
        original = self._write_image("train/NORMAL/source.png", value=60)
        shutil.copyfile(original, self.root / "test/PNEUMONIA/copied.png")

        result = audit_dataset(self.root)

        self.assertEqual(len(result.cross_split_duplicates), 1)
        self.assertEqual(len(result.conflicting_label_duplicates), 1)
        group = result.cross_split_duplicates[0]
        self.assertEqual(group.splits, ("test", "train"))
        self.assertEqual(group.classes, ("NORMAL", "PNEUMONIA"))

    def test_duplicate_group_can_be_within_split_and_cross_split(self) -> None:
        original = self._write_image("train/NORMAL/source.png", value=65)
        shutil.copyfile(original, self.root / "train/NORMAL/local-copy.png")
        shutil.copyfile(original, self.root / "val/NORMAL/cross-copy.png")

        result = audit_dataset(self.root)

        self.assertEqual(len(result.duplicate_groups), 1)
        self.assertEqual(len(result.within_split_duplicates), 1)
        self.assertEqual(len(result.cross_split_duplicates), 1)
        self.assertEqual(len(result.within_split_duplicates[0].records), 2)

    def test_corrupt_and_zero_byte_images_are_reported(self) -> None:
        (self.root / "train/NORMAL/corrupt.jpg").write_bytes(b"not an image")
        (self.root / "test/PNEUMONIA/empty.png").touch()

        result = audit_dataset(self.root)

        self.assertEqual(len(result.unreadable_records), 2)
        reasons = {record.suspicious_reason for record in result.unreadable_records}
        self.assertIn("zero-byte file", reasons)
        self.assertTrue(all(not record.readable for record in result.unreadable_records))

    def test_missing_dataset_path_raises_clear_error(self) -> None:
        missing = self.root.parent / "does-not-exist"

        with self.assertRaisesRegex(DatasetNotFoundError, "does not exist"):
            audit_dataset(missing)

    def test_unexpected_directories_files_and_extensions_are_reported(self) -> None:
        self._write_image("train/NORMAL/valid.png", value=70)
        (self.root / "README.txt").write_text("unexpected", encoding="utf-8")
        (self.root / "extra").mkdir()
        (self.root / "train/UNKNOWN").mkdir()
        (self.root / "train/PNEUMONIA/notes.txt").write_text("notes", encoding="utf-8")

        result = audit_dataset(self.root)
        issue_types = {issue.issue_type for issue in result.structure_issues}

        self.assertIn("file_outside_split", issue_types)
        self.assertIn("unexpected_directory", issue_types)
        self.assertIn("unexpected_class", issue_types)
        self.assertIn("unsupported_extension", issue_types)
        self.assertIn("empty_class_directory", issue_types)

    def test_obsolete_optional_reports_are_removed(self) -> None:
        value = 1
        for split in ("train", "val", "test"):
            for class_name in ("NORMAL", "PNEUMONIA"):
                self._write_image(f"{split}/{class_name}/image.png", value=value)
                value += 1
        result = audit_dataset(self.root)
        output = Path(self.temporary_directory.name) / "audit-output"
        output.mkdir()
        stale_names = (
            "cross_split_duplicates.csv",
            "corrupt_files.csv",
            "structure_issues.csv",
        )
        for name in stale_names:
            (output / name).write_text("stale", encoding="utf-8")

        write_audit_artifacts(result, output)

        for name in stale_names:
            self.assertFalse((output / name).exists())
        summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(Path(summary["dataset_root"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
