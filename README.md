# Pneumonia Screening Research Prototype via HOG-SVM and Transfer-Learned CNNs

## Project overview

This repository contains an educational Machine Learning and Deep Learning research prototype that will compare handcrafted chest X-ray representations with transfer-learned convolutional neural network representations for binary pneumonia screening research.

The classical and CNN pipelines are implemented. The repository does not bundle the full raw dataset, so this checkout contains no reproducible held-out performance result and makes no accuracy claim.

## Research question

How do handcrafted HOG/LBP image representations with kernel-based SVM classifiers compare with transfer-learned CNN representations for binary pneumonia classification from chest X-rays?

## Planned Phase 1 — Classical machine learning

Phase 1 is **PLANNED** and will establish classical baselines using:

- HOG with a Linear SVM
- HOG with an RBF SVM
- HOG plus optional LBP features with an SVM

Any feature scaling, dimensionality reduction, class weighting, and hyperparameter selection will be fitted or selected without using the final test set.

## Planned Phase 2 — Deep learning

Phase 2 is **PLANNED** and will use ImageNet-pretrained transfer learning with:

- ResNet18
- MobileNetV2

The planned sequence is to train a replaced classification head with a frozen backbone, then assess whether fine-tuning later layers with a smaller learning rate is justified. Validation data will guide model selection; test data will remain reserved for final evaluation.

## Optional hybrid approach

The hybrid experiment is **PLANNED AND OPTIONAL**. It will only be considered after the classical ML and transfer-learning baselines work correctly. The proposed approach concatenates HOG/LBP features with CNN embeddings before fitting a classifier such as an SVM.

## Planned evaluation

Executed models will be evaluated using more than accuracy. Planned metrics and analyses include:

- Accuracy
- Precision
- Pneumonia recall / sensitivity
- Specificity
- F1 score
- ROC-AUC
- Confusion matrix
- Probability calibration and calibration curves where applicable

`PNEUMONIA` will be treated consistently as the positive class. No experimental results currently exist, and none are claimed in this repository.

## Repository structure

```text
.
├── README.md              # Project overview and setup guidance
├── requirements.txt       # Minimal planned Python dependency stack
├── configs/
│   └── config.yaml        # Shared non-model project configuration
├── data/
│   ├── raw/               # Unmodified dataset files (Git-ignored)
│   ├── processed/         # Derived dataset files (Git-ignored)
│   └── README.md          # Dataset handling documentation
├── notebooks/             # Exploration and explanation notebooks
├── src/
│   ├── data/              # Reusable dataset-audit and data pipeline code
│   ├── features/          # Reusable feature extraction code
│   ├── models/            # Reusable model code
│   ├── evaluation/        # Metrics and evaluation code
│   └── utils/             # Shared utilities
├── scripts/               # Command-line workflow entry points
├── results/
│   ├── figures/           # Generated figures (Git-ignored)
│   ├── metrics/           # Lightweight result summaries
│   └── models/            # Model artifacts (Git-ignored)
└── tests/                 # Automated tests
```

## Setup

Python environment creation and dependency installation are user-controlled steps. When setup is desired, a typical local workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are installed only through a user-controlled environment setup. Dataset placement and the read-only audit workflow are documented in `data/README.md`.

To view the dataset-audit command options:

```bash
python3 scripts/audit_dataset.py --help
```

The audit is a forensic inventory and integrity check, not preprocessing. It does not resize, normalize, augment, relocate, or delete source images.

## Current project status

**Phase 1 classical feature engineering pipeline implemented; full experimental verification remains pending.** The stored historical audit describes 5,856 source images: 5,216 original training, 16 original validation, and 624 original test images. The current 5,840-row modeling manifest excludes the undersized original 16-image validation folder, partitions the 5,216 original training images into 4,432 train and 784 validation rows, and retains all 624 original test rows. SHA-256-identical training images are kept in the same generated partition, and the stored manifest has no exact-hash overlap between splits. This does **not** prove patient-level separation; the available metadata does not provide stable patient identifiers. The preprocessing implementation standardizes images to single-channel grayscale at 224×224 pixels.

The HOG and LBP feature extraction pipeline (`src/features/`) is implemented and tested:

- **HOG** (`src/features/hog.py`): `skimage.feature.hog` — configurable `orientations`, `pixels_per_cell`, `cells_per_block`, `block_norm` via `configs/config.yaml`. Defaults: 9 orientations, 16×16 pixels-per-cell, 2×2 cells-per-block, L2-Hys → **6,084-dim** vector per image.
- **LBP** (`src/features/lbp.py`): `skimage.feature.local_binary_pattern` — P=8, R=1, method='uniform' → **10-bin** normalised histogram per image (captures diffuse lung opacities that HOG misses).
- **Pipeline** (`src/features/build_features.py`): three feature configs (`hog`, `lbp`, `combined`). `StandardScaler` fitted **strictly on training data only**; val and test are transformed only — never refitted. Outputs cached as `.npz` + `.joblib` under `data/processed/`.

To extract features once `split_manifest.csv` exists:

```bash
# All three feature configs (default params from configs/config.yaml)
python scripts/extract_features.py

# Fine HOG experiment with 8×8 pixels-per-cell
python scripts/extract_features.py --config hog --pixels-per-cell 8 8

# Single config only
python scripts/extract_features.py --config combined
```

The Phase 1 SVM training runner is implemented in `src/models/classical.py` and
`scripts/train_classical.py`. It runs stratified 5-fold cross-validation on the
training split, selects parameters by balanced accuracy, reports a validation
balanced-accuracy score, and serializes each selected estimator under
`results/models/`. The runner selects parameters inside training, fits the base
estimator on training only, and reserves validation for Platt calibration. It
does not access the test split.
Run it after extracting the feature caches:

```bash
python scripts/train_classical.py
```

The three available models are `hog_linear_svm`, `hog_rbf_svm`, and
`hog_lbp_rbf_svm`. Training summaries, including best parameters and timing,
are written to `results/metrics/classical_training.json`. No SVM experiments
have been run yet because feature caches are not currently present. Official
evaluation now fails closed when real artifacts are missing. ResNet18 training,
early stopping, inference, checkpointing, and Grad-CAM are implemented, but no
CNN training result is claimed. See `EVALUATION_READINESS.md` and
`results/metrics/PROVENANCE.md`.


## Medical and research disclaimer

**This project is an educational research prototype and is not intended for clinical diagnosis or medical decision-making.**

Future dataset performance must not be presented as evidence of real-world clinical performance.
