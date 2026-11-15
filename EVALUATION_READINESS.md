# Evaluation readiness report

Date: 16 November 2026. Scope: educational research prototype, not a medical device.

## Requirements matrix

| Area | Status | Evidence or limitation |
|---|---|---|
| Image preprocessing, HOG, LBP | Implemented and tested | Tests cover resizing, descriptors, dimensions, and feature construction. |
| Train-only standardization | Implemented and tested | The feature builder fits `StandardScaler` on train and transforms validation/test. |
| Linear and RBF SVM definitions | Implemented and tested | Class-balanced estimators and stratified CV exist. |
| Probability calibration | Corrected and tested | Base SVM fits on train only; validation is dedicated to Platt calibration. |
| Official classical evaluation | Implemented, blocked on data | Requires calibrated artifacts and real test caches, checks common labels/order, and records hashes. |
| Legacy reported metrics | Invalid or unverified | See `results/metrics/PROVENANCE.md`; do not present them as experimental results. |
| ResNet18 pipeline | Implemented and CPU smoke-tested | ImageNet normalization, frozen backbone, weighted loss, early stopping, checkpoints, inference, and Grad-CAM exist. |
| ResNet18 experiment | Not run | Only 12 sample images are bundled, not the 5,840 manifest images. No CNN metric or checkpoint is claimed. |
| MobileNetV2 architecture | Implemented but untrained | Model definition and late-layer unfreezing exist. |
| Patient grouping | Unresolved limitation | The manifest lacks a reliable patient identifier. |
| Duplicate isolation | Partially verified | Stored audit reports no exact cross-split duplicates; raw data is absent, so this run could not repeat the audit. |
| Streamlit demo | Implemented | Missing models warn; synthetic benchmarks stay hidden; edge density is labeled as non-attribution. |

## Valid workflow and commands

Fit scalers on train only. Select SVM parameters by CV within train. Fit the base SVM on train and Platt calibration on validation. Use validation loss for CNN early stopping. Evaluate frozen models once on identical test IDs at threshold 0.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/audit_dataset.py --data-dir data/raw/chest_xray
python3 scripts/prepare_data.py
python3 scripts/extract_features.py --config all
python3 scripts/train_classical.py
python3 scripts/evaluate.py
python3 scripts/train_cnn.py --epochs 10 --batch-size 16
python3 scripts/evaluate_cnn.py
python3 -m pytest -q
streamlit run app.py
```

Patient-level separation remains a limitation until trustworthy identifiers exist. Exact hashes do not detect near duplicates.

## Five-to-seven-minute presentation script

**Opening, 30 seconds.** This project compares handcrafted image descriptors with learned CNN representations for educational pneumonia screening research. PNEUMONIA is class 1. The system is not intended for diagnosis.

**Dataset and integrity, 45 seconds.** The manifest describes 5,840 images: 4,432 train, 784 validation, and 624 test. The stored audit reports class imbalance and no exact cross-split duplicate hashes. Raw images are not bundled and patient identifiers are unreliable, so patient-level separation cannot be guaranteed.

**Classical pipeline, 60 seconds.** Each image becomes a 224 by 224 grayscale array. HOG summarizes gradient directions; LBP summarizes local texture. A train-fitted scaler standardizes features. Linear SVM learns a hyperplane; the RBF kernel permits a nonlinear boundary. Balanced weights and stratified cross-validation address imbalance.

**Calibration correction, 45 seconds.** The old quickstart refitted on train plus validation and later calibrated on validation, reusing calibration samples in the base fit. The corrected workflow fits the base on train only, then uses validation only for Platt calibration. Test stays sealed.

**Deep learning, 60 seconds.** ResNet18 begins with ImageNet weights. Grayscale images become three channels and use ImageNet normalization. The backbone stays frozen while the head trains. Validation loss controls early stopping. Late blocks may be unfrozen at a lower learning rate. Grad-CAM is explanatory, not proof of clinical reasoning.

**Results correction, 60 seconds.** The repository's 120-sample comparison came from a synthetic fallback. Another benchmark has no traceable generator. Both are excluded. This environment lacks the full dataset, so I make no accuracy claim and no AML-versus-DL ranking.

**Conclusion, 45 seconds.** The contribution is a controlled comparison framework for handcrafted HOG/LBP and learned ResNet18 representations. It now enforces real artifacts, common held-out samples, calibrated probabilities, checkpoints, and provenance. The next step is a complete-data rerun.

## Live demo

1. Run `python3 -m pytest -q`.
2. Run `python3 scripts/evaluate.py`; show the fail-closed missing-artifact error and unchanged metrics.
3. Run `python3 scripts/train_cnn.py --no-pretrained --epochs 1`; show the missing-image error rather than a false training claim.
4. Start `streamlit run app.py` and demonstrate HOG/LBP on a bundled sample.
5. Show the benchmark provenance warning and ResNet18 checkpoint warning.

## 25 likely viva questions

1. **Why HOG?** It encodes local gradient structure and gives an interpretable baseline.
2. **Why LBP?** It summarizes local texture and complements gradients.
3. **What is L2-Hys?** Normalize a HOG block, clip large values, then renormalize.
4. **Why standardize?** SVM margins and RBF distances depend on feature scale.
5. **Linear SVM objective?** A maximum-margin boundary with hinge-loss penalties.
6. **What does RBF add?** Nonlinear similarity based on squared Euclidean distance.
7. **What do C and gamma control?** Error penalty and kernel locality.
8. **Why balanced accuracy?** It weights both class recalls equally.
9. **Why class weights?** They alter error costs under imbalance.
10. **Why seal test?** Test reuse creates optimistic estimates.
11. **What leaked before?** Calibration samples had entered the base-model refit.
12. **Platt scaling?** A sigmoid mapping from scores to probabilities using separate data.
13. **Why Brier score?** It measures squared probability error.
14. **Sensitivity versus specificity?** Pneumonia detection versus correct normal rejection.
15. **Why ResNet18?** It is established and modest in compute.
16. **Why three channels?** Pretrained ImageNet models expect RGB-shaped input.
17. **Why ImageNet normalization?** It matches pretrained-weight expectations.
18. **Why freeze first?** It reduces trainable parameters and overfitting risk.
19. **When fine-tune?** After head convergence, if validation supports it.
20. **What does Grad-CAM show?** Regions influencing a logit through feature-map gradients.
21. **What does it not prove?** Causality, localization accuracy, or clinical validity.
22. **Were legacy metrics real?** One is synthetic; another lacks provenance.
23. **Patient isolation guaranteed?** No, reliable patient IDs are absent.
24. **Is this a hybrid model?** HOG+LBP is feature fusion; CNN fusion is not claimed.
25. **Next valid experiment?** Restore data, rerun audit/training, then evaluate once on common test IDs.

## Tomorrow checklist

- Full dataset exists, or state the blocker clearly.
- Readiness branch is checked out and tests pass.
- App launches and sample feature views load.
- No legacy number is presented as evidence.
- Test count/prevalence come from real evaluation output.
- Comparison rows share identical test IDs.
- Keep the educational-use disclaimer visible.
- Do not call the prototype clinical grade.
