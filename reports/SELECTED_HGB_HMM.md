# Selected HistGradientBoosting: classifier alone versus HMM

The team selected HistGradientBoosting. The application uses the existing optimized artifact as its default. This choice follows the team instruction; it is not a claim that the student-test comparison establishes the best cross-validated classifier.

## Full development comparison

Both methods use the same saved classifier, GPS observations and candidate road segments. Independent matching chooses the highest-scoring candidate per observation; HMM/Viterbi chooses a sequence using those scores and directed-road transition constraints. No model was retrained or decoder tuned against these test trips.

| Metric | Classifier alone | Classifier + HMM | Change |
|---|---:|---:|---:|
| Top-1, all 19,160 observations | 81.92% | 79.49% | -2.43 percentage points |
| Eligible-transition continuity | 85.25% | 98.27% | +13.02 percentage points |
| Mean trip road-arc Macro F1 | 0.6703 | 0.7837 | +0.1134 |
| Exactly matched whole trips | 0 | 0 | 0 |

Top-1 includes candidate misses as errors. Road-arc Macro F1 is not candidate binary F1. Reference labels come from ordered geometric alignment and can be ambiguous.

## Per-trip evidence

| Trip | Observations | Alone Top-1 | HMM Top-1 | Change (percentage points) |
|---|---:|---:|---:|---:|
| 00000000/00000000 | 2503 | 92.41% | 93.93% | +1.52 |
| 00000007/00000007 | 1354 | 91.95% | 92.98% | +1.03 |
| 00000009/00000009 | 1049 | 82.75% | 88.18% | +5.43 |
| 00000040/00000040 | 2995 | 87.61% | 86.21% | -1.40 |
| 00000052/00000052 | 262 | 94.27% | 99.62% | +5.34 |
| 00000056/00000056 | 3958 | 85.95% | 86.81% | +0.86 |
| 00000065/00000065 | 768 | 79.82% | 75.00% | -4.82 |
| 00000076/00000076 | 1486 | 78.53% | 82.30% | +3.77 |
| 00000077/00000077 | 3911 | 64.02% | 48.66% | -15.37 |
| 00000091/00000091 | 874 | 81.46% | 81.69% | +0.23 |

## What the decrease means

HMM improves continuity but does not improve pooled pointwise accuracy with this configuration. A connected directed path can force the sequence away from individually high-scoring candidates. The existing transition model penalizes network/GPS distance mismatch and rejects travel above 60 m/s or across a gap over 30 seconds. At intersections, parallel roads, projection jitter and ambiguous reference alignment, continuity and pointwise labels need not agree.

All seven classifiers showed the same 334 route breaks: 329 speed-limit, two disconnected-component, one road-direction and two timestamp-gap breaks. With finite emission potentials, changing the classifier does not create a missing directed path or relax the speed budget. These are explicit discontinuities, not missing lines that should be filled artificially.

Largest measured net losses in correctly assigned observations: 00000077/00000077: -601, 00000040/00000040: -42, 00000065/00000065: -37.
Net change across all ten trips: -465 correctly assigned observations.

This evidence does not identify one proven cause of every wrong assignment. Further improvement would require grouped training validation of transition assumptions and reference alignment. The reported aggregate result must remain visible; selecting a good-looking single trip is not a substitute.

## Review demonstration

Open the separate UI at http://localhost:8080 and connect it to Laptop 1. The aggregate panel shows the table above. Select trip 00000052/00000052 and enable Reference evaluation to demonstrate live inference. Its Top-1 improves from 94.27% to 99.62%, while the full dataset result decreases. Toggle purple independent segments and green reconstructed sections on the map.

See [two-laptop instructions](../route_recovery/DEPLOYMENT.md) and [local connection evidence](local_separate_ui_api_check.json). The latter proves a same-computer HTTP check, not a physical two-laptop run.

## Reproduce and provenance

Run `python -m route_recovery.verify_integration` to regenerate per-model measurements. The selected-model API verifies the artifact SHA-256 before exposing saved aggregate metrics. Prediction never consumes reference labels; optional evaluation runs after inference.

Artifact SHA-256: `003d635c7ff74cf09e24439985bbf0a5ce778ee0da5a83a8d6007064f3aa51fa`.
Source: [full selected-model measurements](integrated_models/hist_gradient_boosting_optimized.json).
