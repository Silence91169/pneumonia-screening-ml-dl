# Pneumonia Screening Research Prototype via HOG-SVM and Transfer-Learned CNNs

## Project overview

This repository contains an educational Machine Learning and Deep Learning research prototype that will compare handcrafted chest X-ray representations with transfer-learned convolutional neural network representations for binary pneumonia screening research.

All models, experiments, and results described below are **PLANNED** unless explicitly marked otherwise. No dataset has been downloaded and no model has been implemented or evaluated.

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
├── AGENTS.md              # Governing project and research instructions
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
│   ├── data/              # Reusable data pipeline code
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

Dependencies have not been installed by this initialization task. Dataset setup is documented in `data/README.md` and must be performed separately in a later, explicitly authorized task.

## Current project status

**Foundation initialized.** Repository directories, documentation, a shared configuration file, package markers, ignore rules, and the minimal dependency declaration are present. Dataset acquisition, auditing, preprocessing, feature extraction, models, training, evaluation, explainability, and results are all **PLANNED** and have not been implemented.

## Medical and research disclaimer

**This project is an educational research prototype and is not intended for clinical diagnosis or medical decision-making.**

Future dataset performance must not be presented as evidence of real-world clinical performance.

