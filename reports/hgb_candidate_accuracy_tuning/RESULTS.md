# HGB accuracy tuning results

Two training-validation searches were performed. Existing team models were preserved. The new candidate-accuracy model improves over the previous F1-tuned artifact, but does not beat the original untuned baseline on the development partition.

| Model | Candidate accuracy | Road Top-1 | Candidate Macro F1 | Log loss |
|---|---:|---:|---:|---:|
| Untuned HGB | 93.76% | 82.10% | 0.8861 | 0.1519 |
| Previous F1 tuned | 90.97% | 81.92% | 0.8593 | 0.2158 |
| New accuracy tuned | 93.74% | 82.02% | 0.8858 | 0.1523 |

Gain over the previous tuned artifact: +2.77 percentage points candidate accuracy and +0.10 points road Top-1.
Difference from the untuned baseline: -0.02 points candidate accuracy and -0.08 points road Top-1.

## Protocol

Search 1: ten trials, three TPE startup trials, followed by seven adaptive trials; objective grouped road Top-1. It produced a small CV gain but worse development results, and was not promoted.
Search 2: eight trials, three startup trials, followed by five adaptive trials; objective candidate accuracy at probability threshold 0.5. Unweighted recipes were studied because the first training-CV experiment favored them for candidate accuracy. The winning trial was selected before its development evaluation.
Both use the same 20 complete training trips and three GroupKFold splits, then refit on all 80 training trips (1,103,062 candidate rows). No student-test rows enter fitting or trial scoring. Student-test has been reused for development comparisons and is not an untouched final test. The reserved evaluator partition is unused.

Grouped CV candidate accuracy: 93.0616% baseline -> 93.0745% selected.
Grouped CV road Top-1: 79.4205% -> 79.4448%.
Grouped CV Macro F1: 0.876428 -> 0.876217.
The validation gains are very small; no statistical significance is established. Candidate accuracy improved while CV Macro F1 slightly decreased.

## Saved settings

```json
{
  "learning_rate": 0.11919996314024575,
  "max_iter": 100,
  "max_leaf_nodes": 26,
  "min_samples_leaf": 21,
  "l2_regularization": 3.7807877783494277,
  "class_weight": null
}
```

Artifact: `outputs/optimization/hgb_fast/hist_gradient_boosting_candidate_tuned.joblib`. Existing artifacts are unchanged.
The new model is exposed as `hist_gradient_boosting_candidate_tuned` / HistGradientBoosting (accuracy tuned) after restarting the API. The selected default remains the previous artifact, so the historical HMM comparison remains correctly attributed.

A real trip-52 /visualize test verified loading, features, HMM, optional evaluation and map generation; see api_check.json. The all-ten-trip accuracy values above are classifier-only. No pooled HMM improvement is claimed for the new model.

## Reproduce

`python -m experiments.tune_hgb_accuracy`
`python -m experiments.tune_hgb_candidate_accuracy`
`python -m experiments.report_hgb_accuracy_tuning`

Trial logs, results and the API check are in this directory and ../hgb_accuracy_tuning/. Cache files stay under ignored outputs/. All observations, including candidate misses, count in Top-1.
