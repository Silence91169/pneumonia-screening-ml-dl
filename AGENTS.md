# Project 14 — Pneumonia Screening Research Prototype

## 1. Project Title

Pneumonia Screening Research Prototype via HOG-SVM and Transfer-Learned CNNs

## 2. Purpose

This is an educational Machine Learning and Deep Learning research project.

The objective is to compare:

1. Handcrafted image features with classical ML
   - HOG
   - Optional LBP
   - Linear SVM
   - RBF SVM

2. Transfer-learned CNN representations
   - ResNet18
   - MobileNetV2

3. Optional hybrid approach
   - HOG/LBP handcrafted features
   - CNN embeddings
   - Combined representation followed by a classifier such as SVM

This project is NOT a medical diagnostic system.

Never describe model predictions as clinical diagnoses.

---

# 3. Core Research Question

How do handcrafted HOG/LBP image representations with kernel-based SVM
classifiers compare with transfer-learned CNN representations for binary
pneumonia classification from chest X-rays?

---

# 4. Research Motivation

Classical ML uses fixed handcrafted descriptors such as HOG and LBP.

An RBF kernel can transform the decision space of these features, but it
cannot recover image information that was discarded during handcrafted
feature extraction.

Transfer-learned CNNs can instead learn/use hierarchical image
representations containing edges, textures, spatial patterns and
higher-level contextual features.

The project should experimentally investigate this difference.

---

# 5. Dataset

Recommended dataset:

Chest X-Ray Images (Pneumonia) dataset from Kaggle.

Classes:

- NORMAL
- PNEUMONIA

Before model development, perform a dataset audit.

Investigate:

- number of images
- class distribution
- existing train/validation/test structure
- image dimensions
- image formats
- corrupt/unreadable images
- duplicate or near-duplicate images where practical
- possible leakage
- available patient identifiers
- obvious dataset artifacts

If patient identifiers are available, prefer patient-level separation.

If patient identifiers are not available, preserve an appropriate
published dataset partition and clearly document this limitation.

Never silently mix train, validation and test data.

---

# 6. Data Leakage Rules

Data leakage prevention is mandatory.

The test set must not influence:

- preprocessing parameter estimation
- feature selection
- PCA fitting
- scaler fitting
- hyperparameter tuning
- model selection
- early stopping

Fit learned preprocessing operations using training data only.

Validation data may be used for model selection.

Test data should be reserved for final evaluation.

Augmentation must be applied to training data only.

---

# 7. Reproducibility

Use deterministic/reproducible behavior where practical.

Maintain configurable random seeds.

Record important experiment parameters including:

- random seed
- image size
- batch size
- learning rate
- optimizer
- epochs
- model
- frozen/unfrozen layers
- SVM C
- SVM gamma
- HOG configuration
- LBP configuration
- class weighting

Avoid hard-coding machine-specific absolute paths.

---

# 8. Phase 1 — Classical Machine Learning

## Baseline

Pipeline:

Chest X-Ray
→ preprocessing
→ HOG
→ Linear SVM
→ prediction

## Advanced ML experiments

Investigate:

- HOG + Linear SVM
- HOG + RBF SVM
- HOG + LBP + SVM

Optional experiments where justified:

- feature scaling
- PCA
- class weighting
- HOG parameter tuning

HOG parameters may include:

- orientations
- pixels_per_cell
- cells_per_block
- block normalization

SVM tuning may include:

- C
- gamma
- kernel

Hyperparameter tuning must not use the final test set.

---

# 9. Phase 2 — Deep Learning

Use transfer learning.

Primary architectures:

- ResNet18
- MobileNetV2

Start with pretrained ImageNet weights.

Experiment sequence:

1. Replace classification head.
2. Freeze feature extractor/backbone.
3. Train classification head.
4. Evaluate on validation data.
5. If justified, unfreeze late layers.
6. Fine-tune using a smaller learning rate.
7. Compare frozen and fine-tuned performance.

Investigate class imbalance handling where appropriate.

Potential techniques:

- class-weighted loss
- weighted sampling

Do not automatically use both.
Experiment and document the choice.

---

# 10. Optional Hybrid Experiment

Only implement this after the ML and DL baselines work correctly.

Possible architecture:

X-Ray
├── HOG/LBP features
└── CNN embedding

Then:

HOG/LBP features + CNN embedding
→ concatenate
→ optional scaling/dimensionality reduction
→ SVM/classifier
→ prediction

The hybrid model must use the same data-splitting and leakage-prevention
rules as all other experiments.

---

# 11. Evaluation

Do NOT evaluate models using accuracy alone.

Report at minimum:

- Accuracy
- Precision
- Pneumonia Recall / Sensitivity
- Specificity
- F1 Score
- ROC-AUC
- Confusion Matrix

Also investigate:

- probability calibration
- calibration curve

Where probability outputs are compared, ensure the method used to
produce probabilities is documented.

The positive class should be defined consistently as PNEUMONIA.

Important formulas:

Sensitivity / Recall:

TP / (TP + FN)

Specificity:

TN / (TN + FP)

Precision:

TP / (TP + FP)

F1:

2 * Precision * Recall / (Precision + Recall)

---

# 12. Explainability

For CNN models, generate Grad-CAM or a comparable saliency technique.

Include representative examples such as:

- correctly classified NORMAL
- correctly classified PNEUMONIA
- false positive
- false negative

Do not interpret saliency maps as proof of clinical reasoning.

For HOG, provide HOG visualizations where useful.

---

# 13. Fair Comparison

ML and DL experiments should use comparable data partitions.

Do not manipulate test sets to improve results.

Maintain a central experiment-results table.

Target structure:

| Model | Recall | Specificity | Precision | F1 | ROC-AUC |
|-------|--------|-------------|-----------|----|---------|
| HOG + Linear SVM | | | | | |
| HOG + RBF SVM | | | | | |
| HOG + LBP + SVM | | | | | |
| ResNet18 Frozen | | | | | |
| ResNet18 Fine-tuned | | | | | |
| MobileNetV2 | | | | | |
| Hybrid | | | | | |

Only include models that were actually executed.

Never fabricate experimental results.

---

# 14. Team Responsibilities

## Shivam — Data & Experimental Pipeline

Primary responsibilities:

- dataset audit
- duplicate investigation
- preprocessing
- splitting strategy
- augmentation
- leakage prevention
- reproducible data pipeline
- shared DataLoaders/dataset utilities

## Yash — Classical ML

Primary responsibilities:

- HOG
- LBP
- feature engineering
- Linear SVM
- RBF SVM
- hyperparameter tuning
- optional PCA
- classical ML experiments

## Shitanshu — Deep Learning

Primary responsibilities:

- ResNet18
- MobileNetV2
- transfer learning
- frozen-backbone experiments
- fine-tuning
- training pipeline
- CNN embeddings
- DL experiments

## Keshav — Evaluation

Primary responsibilities:

- common metric implementation
- confusion matrices
- ROC curves
- calibration
- evaluation consistency
- result aggregation
- explainability support

## Kavya — Research & Integration

Primary responsibilities:

- literature review
- experiment comparison
- research-gap discussion
- result interpretation
- limitations
- documentation
- presentation integration

Team boundaries are organizational, not strict code ownership boundaries.

---

# 15. Repository Architecture

Target structure:

pneumonia-screening/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── tests/

Do not place unnecessary code in notebooks.

Reusable functionality should live under src/.

Notebooks should primarily be used for:

- exploration
- visualization
- experiment explanation

---

# 16. Git Rules

Do NOT commit:

- raw dataset images
- virtual environments
- Python caches
- large trained checkpoints unless explicitly required
- credentials
- Kaggle API keys
- secrets

Use meaningful commits.

Prefer small, reviewable changes.

Do not overwrite working implementations without justification.

---

# 17. Coding Standards

Use Python.

Primary libraries:

- NumPy
- pandas
- matplotlib
- OpenCV
- scikit-image
- scikit-learn
- PyTorch
- torchvision

Code should be:

- modular
- readable
- documented
- reproducible
- configurable

Use type hints where they improve clarity.

Avoid unnecessary abstractions.

Do not introduce new dependencies unless justified.

---

# 18. Development Rules for Codex

IMPORTANT:

Do not attempt to build the entire project in one task.

Work incrementally.

Before implementing a major feature:

1. inspect existing repository code
2. understand current architecture
3. identify affected files
4. implement the smallest complete change
5. test it
6. report what changed

Do not silently change experimental methodology.

Do not fabricate datasets, results or metrics.

Do not claim code was tested if it was not executed.

If an assumption could affect scientific validity, explicitly state it.

Prefer fixing the underlying problem rather than hiding errors.

---

# 19. Medical/Research Disclaimer

Every final report and application must clearly communicate:

"This project is an educational research prototype and is not intended
for clinical diagnosis or medical decision-making."

Dataset performance must not be presented as evidence of real-world
clinical performance.
