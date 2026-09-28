# Dataset location and audit

This project expects the **Chest X-Ray Images (Pneumonia)** dataset commonly distributed through Kaggle. Obtain it manually from the dataset provider and review its license and provenance before use. Do not place Kaggle credentials in this repository.

## Required location and structure

Extract or place the dataset at `data/raw/chest_xray/` so that it resembles:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

`validation` or `valid` is also recognized as the validation split. Split and class names are matched case-insensitively, but the expected classes remain exactly `NORMAL` and `PNEUMONIA`; `PNEUMONIA` is the positive class.

Raw files must remain unchanged. Do not resize, normalize, augment, move, rename, or delete source images during the audit. Dataset contents under `data/raw/` and derived files under `data/processed/` are excluded from Git. This tracked README is the only repository documentation stored alongside those directories.

**Current status:** The dataset was supplied manually at `data/raw/chest_xray/` and has been audited. It contains 5,856 readable images. No project task downloaded the dataset automatically.

## Running the read-only audit

After placing the dataset, run from the repository root:

```bash
python3 scripts/audit_dataset.py
```

To audit a dataset stored elsewhere without changing configuration:

```bash
python3 scripts/audit_dataset.py --data-dir /path/to/chest_xray
```

The audit inventories image counts and class percentages; dimensions, modes, channels, extensions, and file sizes; unreadable, zero-byte, and suspicious files; directory/schema problems; and SHA-256 exact duplicates within and across splits. Identical contents with conflicting labels are explicitly flagged. Detailed output is written to `results/dataset_audit/` only after a real dataset is found and audited.

The distributed filenames are not treated as reliable patient identifiers without authoritative metadata. Therefore: **Patient-level separation cannot be guaranteed from the available dataset metadata.** This is a project limitation and must be revisited if reliable source metadata becomes available.

## Audit is not preprocessing

**DATASET AUDIT ≠ PREPROCESSING.** The audit reads source images to characterize and validate them; it does not transform them. Preprocessing and split-policy decisions belong to a later task and must follow the leakage rules in `AGENTS.md`.
