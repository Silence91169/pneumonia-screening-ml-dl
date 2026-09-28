"""Read-only forensic auditing for the chest X-ray dataset.

The functions in this module inventory source files without resizing,
normalizing, moving, deleting, or otherwise modifying them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

from PIL import Image


EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
EXPECTED_SPLITS = ("train", "validation", "test")
SUPPORTED_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
PATIENT_LEVEL_LIMITATION = (
    "Patient-level separation cannot be guaranteed from the available "
    "dataset metadata."
)


class DatasetAuditError(RuntimeError):
    """Base error for dataset-audit failures."""


class DatasetNotFoundError(DatasetAuditError):
    """Raised when the configured dataset directory does not exist."""


class EmptyDatasetError(DatasetAuditError):
    """Raised when no candidate images exist in expected class directories."""


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata collected for one candidate image file."""

    relative_path: str
    split: str
    class_name: str
    extension: str
    width: int | None
    height: int | None
    channel_count: int | None
    image_mode: str | None
    readable: bool
    file_size_bytes: int | None
    sha256: str | None
    suspicious_reason: str | None
    error: str | None


@dataclass(frozen=True)
class StructureIssue:
    """A structural or integrity concern found during discovery."""

    severity: str
    issue_type: str
    relative_path: str
    message: str


@dataclass(frozen=True)
class DuplicateGroup:
    """Files sharing identical SHA-256 content."""

    sha256: str
    records: tuple[ImageMetadata, ...]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({record.split for record in self.records}))

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({record.class_name for record in self.records}))

    @property
    def is_cross_split(self) -> bool:
        return len(self.splits) > 1

    @property
    def has_conflicting_labels(self) -> bool:
        return len(self.classes) > 1


@dataclass(frozen=True)
class AuditResult:
    """Complete in-memory result of one read-only dataset audit."""

    dataset_root: Path
    records: tuple[ImageMetadata, ...]
    structure_issues: tuple[StructureIssue, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    patient_level_status: str = PATIENT_LEVEL_LIMITATION

    @property
    def total_images(self) -> int:
        return len(self.records)

    @property
    def unreadable_records(self) -> tuple[ImageMetadata, ...]:
        return tuple(record for record in self.records if not record.readable)

    @property
    def within_split_duplicates(self) -> tuple[DuplicateGroup, ...]:
        within_split: list[DuplicateGroup] = []
        for group in self.duplicate_groups:
            records_by_split: dict[str, list[ImageMetadata]] = defaultdict(list)
            for record in group.records:
                records_by_split[record.split].append(record)
            within_split.extend(
                DuplicateGroup(group.sha256, tuple(records))
                for _, records in sorted(records_by_split.items())
                if len(records) > 1
            )
        return tuple(within_split)

    @property
    def cross_split_duplicates(self) -> tuple[DuplicateGroup, ...]:
        return tuple(group for group in self.duplicate_groups if group.is_cross_split)

    @property
    def conflicting_label_duplicates(self) -> tuple[DuplicateGroup, ...]:
        return tuple(
            group for group in self.duplicate_groups if group.has_conflicting_labels
        )

    def split_counts(self) -> dict[str, int]:
        counts = Counter(record.split for record in self.records)
        return {split: counts.get(split, 0) for split in EXPECTED_SPLITS}

    def class_counts(self, split: str | None = None) -> dict[str, int]:
        records = self.records
        if split is not None:
            records = tuple(record for record in records if record.split == split)
        counts = Counter(record.class_name for record in records)
        return {class_name: counts.get(class_name, 0) for class_name in EXPECTED_CLASSES}

    def dimension_summary(self) -> dict[str, object]:
        readable = [
            record
            for record in self.records
            if record.readable and record.width is not None and record.height is not None
        ]
        if not readable:
            return {
                "readable_image_count": 0,
                "minimum_width": None,
                "maximum_width": None,
                "median_width": None,
                "minimum_height": None,
                "maximum_height": None,
                "median_height": None,
                "unique_dimension_count": 0,
                "image_modes": {},
                "channel_counts": {},
                "extensions": dict(sorted(Counter(r.extension for r in self.records).items())),
            }

        widths = [record.width for record in readable if record.width is not None]
        heights = [record.height for record in readable if record.height is not None]
        return {
            "readable_image_count": len(readable),
            "minimum_width": min(widths),
            "maximum_width": max(widths),
            "median_width": median(widths),
            "minimum_height": min(heights),
            "maximum_height": max(heights),
            "median_height": median(heights),
            "unique_dimension_count": len({(r.width, r.height) for r in readable}),
            "image_modes": dict(sorted(Counter(r.image_mode for r in readable).items())),
            "channel_counts": dict(
                sorted(Counter(str(r.channel_count) for r in readable).items())
            ),
            "extensions": dict(sorted(Counter(r.extension for r in self.records).items())),
        }


def _canonical_split(name: str) -> str | None:
    return SPLIT_ALIASES.get(name.casefold())


def _canonical_class(name: str, expected_classes: Sequence[str]) -> str | None:
    lookup = {class_name.casefold(): class_name for class_name in expected_classes}
    return lookup.get(name.casefold())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path, root: Path, split: str, class_name: str) -> ImageMetadata:
    relative_path = path.relative_to(root).as_posix()
    extension = path.suffix.casefold()
    size: int | None = None
    digest: str | None = None
    suspicious_reason: str | None = None

    try:
        size = path.stat().st_size
        if size == 0:
            suspicious_reason = "zero-byte file"
        elif size < 128:
            suspicious_reason = "very small file (<128 bytes)"
        digest = _sha256(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            channel_count = len(image.getbands())
            image_mode = (
                "GRAYSCALE" if image.mode in {"1", "L", "I", "F"} else image.mode
            )
        return ImageMetadata(
            relative_path,
            split,
            class_name,
            extension,
            width,
            height,
            channel_count,
            image_mode,
            True,
            size,
            digest,
            suspicious_reason,
            None,
        )
    except (OSError, ValueError) as exc:
        return ImageMetadata(
            relative_path,
            split,
            class_name,
            extension,
            None,
            None,
            None,
            None,
            False,
            size,
            digest,
            suspicious_reason or "decoder could not read image",
            f"{type(exc).__name__}: {exc}",
        )


def _find_duplicates(records: Iterable[ImageMetadata]) -> tuple[DuplicateGroup, ...]:
    by_hash: dict[str, list[ImageMetadata]] = defaultdict(list)
    for record in records:
        if record.sha256:
            by_hash[record.sha256].append(record)
    return tuple(
        DuplicateGroup(sha256=digest, records=tuple(sorted(group, key=lambda r: r.relative_path)))
        for digest, group in sorted(by_hash.items())
        if len(group) > 1
    )


def audit_dataset(
    dataset_root: str | Path,
    expected_classes: Sequence[str] = EXPECTED_CLASSES,
) -> AuditResult:
    """Audit a dataset tree without modifying any source file.

    Args:
        dataset_root: Root containing train/validation/test directories.
        expected_classes: Required binary class directory names.

    Raises:
        DatasetNotFoundError: If the root is absent or is not a directory.
        EmptyDatasetError: If no supported candidate images are found.
    """

    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise DatasetNotFoundError(
            f"Dataset directory does not exist: {root}. Place the dataset there "
            "or pass --data-dir. No files were changed."
        )
    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset path is not a directory: {root}")

    issues: list[StructureIssue] = []
    candidates: list[tuple[Path, str, str]] = []
    discovered_splits: dict[str, Path] = {}

    for entry in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not entry.is_dir():
            issues.append(
                StructureIssue(
                    "warning",
                    "file_outside_split",
                    entry.relative_to(root).as_posix(),
                    "File is outside an expected split/class directory.",
                )
            )
            continue
        split = _canonical_split(entry.name)
        if split is None:
            issues.append(
                StructureIssue(
                    "warning",
                    "unexpected_directory",
                    entry.relative_to(root).as_posix(),
                    "Directory name is not a recognized dataset split.",
                )
            )
            continue
        if split in discovered_splits:
            issues.append(
                StructureIssue(
                    "error",
                    "duplicate_split_alias",
                    entry.relative_to(root).as_posix(),
                    f"Multiple directories map to canonical split '{split}'.",
                )
            )
            continue
        discovered_splits[split] = entry

    for expected_split in EXPECTED_SPLITS:
        if expected_split not in discovered_splits:
            issues.append(
                StructureIssue(
                    "error",
                    "missing_split",
                    ".",
                    f"Required split '{expected_split}' was not found.",
                )
            )

    for split, split_path in sorted(discovered_splits.items()):
        discovered_classes: dict[str, Path] = {}
        for entry in sorted(split_path.iterdir(), key=lambda item: item.name.casefold()):
            relative = entry.relative_to(root).as_posix()
            if not entry.is_dir():
                issues.append(
                    StructureIssue(
                        "warning",
                        "file_outside_class",
                        relative,
                        "File is directly under a split rather than a class directory.",
                    )
                )
                continue
            class_name = _canonical_class(entry.name, expected_classes)
            if class_name is None:
                issues.append(
                    StructureIssue(
                        "error",
                        "unexpected_class",
                        relative,
                        f"Expected class names: {', '.join(expected_classes)}.",
                    )
                )
                continue
            if class_name in discovered_classes:
                issues.append(
                    StructureIssue(
                        "error",
                        "duplicate_class_alias",
                        relative,
                        f"Multiple directories map to class '{class_name}'.",
                    )
                )
                continue
            discovered_classes[class_name] = entry

        for class_name in expected_classes:
            if class_name not in discovered_classes:
                issues.append(
                    StructureIssue(
                        "error",
                        "missing_class",
                        split_path.relative_to(root).as_posix(),
                        f"Class directory '{class_name}' is missing from split '{split}'.",
                    )
                )

        for class_name, class_path in sorted(discovered_classes.items()):
            class_files = [path for path in class_path.rglob("*") if path.is_file()]
            if not class_files:
                issues.append(
                    StructureIssue(
                        "warning",
                        "empty_class_directory",
                        class_path.relative_to(root).as_posix(),
                        "Class directory contains no files.",
                    )
                )
            for path in sorted(class_files):
                relative = path.relative_to(root).as_posix()
                if path.parent != class_path:
                    issues.append(
                        StructureIssue(
                            "warning",
                            "unexpected_nested_file",
                            relative,
                            "Image is nested below the expected split/class level.",
                        )
                    )
                if path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                    issues.append(
                        StructureIssue(
                            "warning",
                            "unsupported_extension",
                            relative,
                            f"Unsupported extension '{path.suffix or '<none>'}'.",
                        )
                    )
                    continue
                candidates.append((path, split, class_name))

    if not candidates:
        raise EmptyDatasetError(
            f"No supported image files were found under {root}. The audit did not "
            "create result files."
        )

    records = tuple(
        _inspect_image(path, root, split, class_name)
        for path, split, class_name in candidates
    )
    return AuditResult(
        dataset_root=root.resolve(),
        records=records,
        structure_issues=tuple(issues),
        duplicate_groups=_find_duplicates(records),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _duplicate_rows(groups: Iterable[DuplicateGroup]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(groups, start=1):
        for record in group.records:
            rows.append(
                {
                    "group_id": group_index,
                    "sha256": group.sha256,
                    "relative_path": record.relative_path,
                    "split": record.split,
                    "class_name": record.class_name,
                    "group_splits": "|".join(group.splits),
                    "group_classes": "|".join(group.classes),
                    "conflicting_labels": group.has_conflicting_labels,
                }
            )
    return rows


def write_audit_artifacts(result: AuditResult, output_dir: str | Path) -> tuple[Path, ...]:
    """Write reports derived from a completed real audit and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_rows: list[dict[str, object]] = []
    overall_counts = result.class_counts()
    for class_name, count in overall_counts.items():
        summary_rows.append(
            {
                "scope": "overall",
                "split": "all",
                "class_name": class_name,
                "count": count,
                "percentage": round(100 * count / result.total_images, 6),
            }
        )
    for split, split_total in result.split_counts().items():
        split_classes = result.class_counts(split)
        for class_name, count in split_classes.items():
            percentage = round(100 * count / split_total, 6) if split_total else 0.0
            summary_rows.append(
                {
                    "scope": "split",
                    "split": split,
                    "class_name": class_name,
                    "count": count,
                    "percentage": percentage,
                }
            )
    summary_path = output / "dataset_summary.csv"
    _write_csv(
        summary_path,
        ("scope", "split", "class_name", "count", "percentage"),
        summary_rows,
    )
    written.append(summary_path)

    metadata_path = output / "image_metadata.csv"
    metadata_fields = tuple(asdict(result.records[0]).keys())
    _write_csv(metadata_path, metadata_fields, (asdict(record) for record in result.records))
    written.append(metadata_path)

    dimension_counts = Counter(
        (record.width, record.height, record.channel_count, record.image_mode)
        for record in result.records
        if record.readable
    )
    dimensions_path = output / "image_dimensions.csv"
    _write_csv(
        dimensions_path,
        ("width", "height", "channel_count", "image_mode", "count"),
        (
            {
                "width": key[0],
                "height": key[1],
                "channel_count": key[2],
                "image_mode": key[3],
                "count": count,
            }
            for key, count in sorted(dimension_counts.items(), key=lambda item: str(item[0]))
        ),
    )
    written.append(dimensions_path)

    duplicate_fields = (
        "group_id",
        "sha256",
        "relative_path",
        "split",
        "class_name",
        "group_splits",
        "group_classes",
        "conflicting_labels",
    )
    if result.within_split_duplicates:
        path = output / "duplicate_report.csv"
        _write_csv(path, duplicate_fields, _duplicate_rows(result.within_split_duplicates))
        written.append(path)
    else:
        (output / "duplicate_report.csv").unlink(missing_ok=True)
    if result.cross_split_duplicates:
        path = output / "cross_split_duplicates.csv"
        _write_csv(path, duplicate_fields, _duplicate_rows(result.cross_split_duplicates))
        written.append(path)
    else:
        (output / "cross_split_duplicates.csv").unlink(missing_ok=True)
    if result.unreadable_records:
        path = output / "corrupt_files.csv"
        _write_csv(
            path,
            metadata_fields,
            (asdict(record) for record in result.unreadable_records),
        )
        written.append(path)
    else:
        (output / "corrupt_files.csv").unlink(missing_ok=True)
    if result.structure_issues:
        path = output / "structure_issues.csv"
        fields = tuple(asdict(result.structure_issues[0]).keys())
        _write_csv(path, fields, (asdict(issue) for issue in result.structure_issues))
        written.append(path)
    else:
        (output / "structure_issues.csv").unlink(missing_ok=True)

    normal_count = overall_counts["NORMAL"]
    pneumonia_count = overall_counts["PNEUMONIA"]
    resolved_output = output.resolve()
    project_root = resolved_output.parents[1] if len(resolved_output.parents) > 1 else None
    try:
        dataset_root = result.dataset_root.relative_to(project_root).as_posix()
    except (TypeError, ValueError):
        dataset_root = result.dataset_root.name
    json_summary = {
        "dataset_root": dataset_root,
        "total_images": result.total_images,
        "split_counts": result.split_counts(),
        "class_counts": overall_counts,
        "pneumonia_to_normal_ratio": (
            pneumonia_count / normal_count if normal_count else None
        ),
        "pneumonia_prevalence": pneumonia_count / result.total_images,
        "image_characteristics": result.dimension_summary(),
        "unreadable_image_count": len(result.unreadable_records),
        "exact_duplicate_group_count": len(result.duplicate_groups),
        "within_split_duplicate_group_count": len(result.within_split_duplicates),
        "cross_split_duplicate_group_count": len(result.cross_split_duplicates),
        "conflicting_label_duplicate_group_count": len(
            result.conflicting_label_duplicates
        ),
        "structure_issue_count": len(result.structure_issues),
        "patient_level_separation": result.patient_level_status,
    }
    json_path = output / "audit_summary.json"
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(json_summary, file_handle, indent=2)
        file_handle.write("\n")
    written.append(json_path)

    return tuple(written)
