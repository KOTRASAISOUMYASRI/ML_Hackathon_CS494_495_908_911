# Member 4 integration status

## Completed after Member 2 and Member 3 merges

- Connected all five classifier families and both optimized variants to HMM
  emissions, API discovery and the UI dropdown. Artifacts remain unchanged.
- Validated all seven artifacts on all ten development student-test trips:
  **70 successful numeric API runs**, plus a map endpoint check for each model.
- Saved per-trip metrics, diagnostic causes, runtimes and artifact SHA-256
  fingerprints in [integration results](integrated_models/SUMMARY.md).
- Confirmed the browser can reconstruct trip 52 with optimized HGB, show the
  map, preserve selections and display optional independent-versus-HMM metrics.
- Added explicit LFS-pointer errors, one-model caching, model labels and separate
  classifier/decoder runtimes. Added the missing purple map-layer legend.
- Added artifact-compatible installation requirements (scikit-learn 1.8.0),
  updated deployment instructions and a reproducible verification command.
- Regression suite: **25 tests passed**, including directed-road behaviour,
  fallback/missing candidates, geometry, CORS, reference isolation, model
  discovery and LFS-pointer rejection.

Optimized HGB has the highest HMM Top-1 in this development comparison (79.49%).
Baseline HGB has the highest independent Top-1 (82.10%). HMM continuity is 98.27%
for all seven models, but pooled pointwise accuracy falls with HMM. No whole
trip is exactly recovered. The team has now selected HistGradientBoosting;
the optimized artifact is the API/UI default. See the
[main before/after report](SELECTED_HGB_HMM.md).

## Final demonstration and connection setup

- A dedicated UI panel shows the selected classifier alone versus HMM, the
  pooled accuracy change and continuity change. Current-trip metrics remain
  separate. Artifact fingerprints protect against displaying stale aggregates.
- Added API/UI PowerShell launch scripts and minimal Laptop 2 dependencies.
- Added a standard-library connection checker, usable on Laptop 2 without ML
  libraries, with received-byte logs and JSON evidence output.
- Verified real HTTP health, trip inventory, CORS preflight and selected-model
  route/map receipt using separate local ports 8000 and 8080. See
  [local evidence](local_separate_ui_api_check.json).
- Verified the browser default, full-dataset panel and a live evaluated trip 52
  through the separate UI. Laptop 1 logged the incoming POST request.
- Complete [two-laptop instructions](../route_recovery/DEPLOYMENT.md) cover
  installation, branch setup, IP addresses, CORS, firewall and demo evidence.

## Remaining boundaries

1. Further decoder tuning is optional improvement work and requires grouped
   training validation; do not tune against these student-test results. The
   implemented HMM's accuracy decrease remains visible.
2. Physical two-laptop verification requires running the documented commands on
   both actual devices. Local two-port checks do not prove Wi-Fi behaviour.
3. Reserved evaluator execution awaits the actual input/reference directories.
4. Re-run verification after teammates replace any model artifact.

The local implementation and integration checks above are complete. This does
not claim perfect route reconstruction or completed hardware deployment.

## Implemented without changing teammates' models

- Fixed trip dropdown selection preservation and snapshotted request inputs.
- Separated UI JavaScript into an external file served by both UI entry points.
- Restored read-only classifier loading, shared feature/class validation and
  candidate-aligned normalised log emission scores.
- Added a server-configured artifact slot for future model handoff.
- Restored optional reference evaluation after inference, with aggregate metrics
  only; reference road IDs and geometry are not returned.
- Added disabled-by-default reserved input/reference configuration.
- Added a development-only batch comparison command and integration tests.

## Measured existing Decision Tree baseline

See [full per-trip results](member4_classifier_hmm.json). All ten student-test
trips and 19,160 observations were evaluated with the current shared reference
alignment. Candidate misses count as errors in Top-1.

| Metric | Independent classifier | Classifier + HMM |
|---|---:|---:|
| All-observation Top-1 | 80.56% | 78.63% |
| Mean per-trip road-arc Macro F1 | 0.6363 | 0.7404 |
| Eligible-transition continuity | 83.09% | 98.27% |
| Exactly matched whole trips | 0 | 0 |

HMM improves connectivity but lowers overall pointwise accuracy. Trip 00000077
falls from 63.90% to 48.02%, outweighing gains elsewhere. Trip 00000052 improves
from 92.75% to 96.56%. These are derived-label comparisons, not verified physical
ground truth. Candidate binary Macro F1/Log Loss are separate metrics stored in
the per-trip results, not the road-arc Macro F1 above.

## Ownership

No model implementation, saved artifact or teammate training script was changed
by the Member 4 integration. The separate inference requirements file pins the
verified library version; the root training requirements remain unchanged.
