# Saved-model integration results

All seven saved classifiers completed the ten student-test trips (19,160 observations each), API reference evaluation and map generation. No training or tuning was performed.

| Model | Independent Top-1 | HMM Top-1 | Independent continuity | HMM continuity | HMM road Macro F1 | Inference seconds |
|---|---:|---:|---:|---:|---:|---:|
| decision_tree | 80.56% | 78.63% | 83.09% | 98.27% | 0.7404 | 34.37 |
| logistic_regression | 80.02% | 78.83% | 83.29% | 98.27% | 0.7660 | 27.05 |
| knn | 79.13% | 78.20% | 81.94% | 98.27% | 0.7332 | 38.48 |
| random_forest | 81.96% | 79.35% | 84.63% | 98.27% | 0.7713 | 31.39 |
| random_forest_optimized | 82.02% | 79.24% | 84.66% | 98.27% | 0.7717 | 35.98 |
| hist_gradient_boosting | 82.10% | 79.47% | 85.18% | 98.27% | 0.7795 | 31.45 |
| hist_gradient_boosting_optimized | 81.92% | 79.49% | 85.25% | 98.27% | 0.7837 | 31.56 |

## Interpretation

- Top-1 includes all observations; candidate misses count as errors. Road Macro F1 is averaged across trips, not candidate binary F1.
- These use derived ordered geometric reference labels. Ambiguous alignment limits the accuracy conclusions.
- HMM increases continuity while lowering pooled pointwise Top-1 for every tested model. No model exactly recovers any whole trip.
- Every model has 334 breaks: 329 speed-limit, two disconnected-component, one road-direction and two timestamp-gap breaks. Finite emission scores do not create missing directed connections. No artificial connector is drawn.
- Optimized HGB has the highest HMM Top-1 here; baseline HGB has the highest independent Top-1. This is development evidence, not a final model selection or proof that tuning universally improved performance.
- Inference seconds sum ten numeric API runs, including preparation, first model load and decoding. Optional reference evaluation and the extra map check are excluded. Models run sequentially on this machine; timings are not deployment benchmarks.
- Select the final model and any decoder parameters using grouped training validation. Do not optimize parameters against this student-test comparison.

## Reproduce

From the repository root: `python -m route_recovery.verify_integration`.
Install `route_recovery/requirements-inference.txt` and download the saved artifacts with `git lfs pull` first.
Per-model JSON contains trip metrics, runtimes, gap diagnostics and the artifact SHA-256. No reserved evaluator data was used.
