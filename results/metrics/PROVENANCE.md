# Metric provenance

Last reviewed: 16 November 2026.

| Artifact | Provenance | Scientific status |
|---|---|---|
| `classical_training.json` | Produced by `scripts/train_classical.py` from the full training and validation feature caches. Records five-fold CV selection, validation scores, fitting times, split use, and artifact paths. | **Verified training record.** Not held-out test evidence. |
| `model_comparison.json` / `.csv` | Produced by `scripts/evaluate.py` from the three calibrated SVM artifacts and the real 624-row held-out feature caches. The evaluator fails closed when required artifacts are absent and records model/cache SHA-256 hashes. | **Real held-out test evidence.** |
| `resnet18_test_metrics.json` | Produced by `scripts/evaluate_cnn.py` from `resnet18_best.pt` and the 624 test rows in `split_manifest.csv`. Contains the evaluation threshold, sample count, sample paths, and metrics. | **Real held-out test evidence.** |
| `resnet18_best.history.json` | Training history written alongside the selected checkpoint by `scripts/train_cnn.py`. The checkpoint records the selected epoch and validation loss. | **Verified training history.** |

## Historical context

An earlier implementation could generate a synthetic 120-sample comparison when real feature artifacts were absent. That fallback was removed, and the current `model_comparison` artifacts replaced it with real 624-image evaluation output. The current evaluator refuses to write results unless all required trained models and real held-out feature caches exist.

The former `test_benchmark_metrics.json` also described 120 samples but had no generating script, sample identifiers, artifact hashes, or run configuration. It was unverified and was removed from the final repository; it must not be used as research evidence.

## Evaluation policy and limitations

`PNEUMONIA` is encoded as the positive class (`1`) and `NORMAL` as the negative class (`0`). The test split is reserved for final evaluation and is not used for fitting, hyperparameter selection, calibration, or checkpoint selection.

The classical feature caches store labels but not sample identifiers. Their 624 labels and ordering agree across the HOG and combined caches and follow the test-manifest workflow, but identifier-level equality cannot be independently reconstructed from those caches alone. The ResNet18 result records all evaluated test paths.
