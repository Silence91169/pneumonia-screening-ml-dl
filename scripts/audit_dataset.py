#!/usr/bin/env python3
"""Command-line entry point for the read-only dataset audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import (  # noqa: E402
    DatasetAuditError,
    audit_dataset,
    write_audit_artifacts,
)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_paths(config_path: Path) -> tuple[Path, Path]:
    try:
        with config_path.open("r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle) or {}
        paths = config["paths"]
        return (
            _resolve_project_path(paths["dataset"]),
            _resolve_project_path(paths["dataset_audit"]),
        )
    except FileNotFoundError as exc:
        raise DatasetAuditError(f"Configuration file not found: {config_path}") from exc
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise DatasetAuditError(
            f"Invalid configuration in {config_path}: expected paths.dataset and "
            "paths.dataset_audit."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the raw chest X-ray dataset without modifying, preprocessing, "
            "or moving source images."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="configuration YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="override the configured dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override the configured audit output directory",
    )
    return parser


def _print_summary(result: object, written: tuple[Path, ...]) -> None:
    overall = result.class_counts()
    split_counts = result.split_counts()
    print("Dataset Audit")
    print("-" * 40)
    print(f"Total images: {result.total_images}")
    print(f"Train: {split_counts['train']}")
    print(f"Validation: {split_counts['validation']}")
    print(f"Test: {split_counts['test']}")
    print("\nClass distribution:")
    print(f"NORMAL: {overall['NORMAL']}")
    print(f"PNEUMONIA: {overall['PNEUMONIA']}")
    print(f"\nUnreadable images: {len(result.unreadable_records)}")
    print(f"Exact duplicate groups: {len(result.duplicate_groups)}")
    print(f"Cross-split duplicate groups: {len(result.cross_split_duplicates)}")
    print(
        "Conflicting-label duplicate groups: "
        f"{len(result.conflicting_label_duplicates)}"
    )
    print(f"Structural issues: {len(result.structure_issues)}")
    print(f"\nPatient-level separation: {result.patient_level_status}")
    print(f"\nAudit artifacts written: {len(written)}")
    if written:
        print(f"Output directory: {written[0].parent}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configured_data, configured_output = _load_paths(
            _resolve_project_path(args.config)
        )
        data_dir = _resolve_project_path(args.data_dir) if args.data_dir else configured_data
        output_dir = (
            _resolve_project_path(args.output_dir)
            if args.output_dir
            else configured_output
        )
        result = audit_dataset(data_dir)
        written = write_audit_artifacts(result, output_dir)
    except DatasetAuditError as exc:
        print(f"Dataset audit could not run: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Dataset audit failed while reading or writing files: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
