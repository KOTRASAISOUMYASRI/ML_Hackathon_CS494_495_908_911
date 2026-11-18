# Member 2 handoff — candidate features

Call `feature_engineering.build_feature_table(trip, labeled=True)` for a labeled
training or student-test trip. It returns a `FeatureResult` with `candidates`
(the model table) and `observations` (alignment and candidate-recall diagnostics).
The CLI in the repository README exports these as `candidates.csv` and
`observations.csv`. Run each partition into a separate folder.

## Using the supplied dataset

- Fit and cross-validate on the archive's 80-trip `train` partition. For CV
  within training, use `trip_id` as the group. Use the separate 10-trip
  `student_test` partition for development evaluation and model comparison.
- GPS cleaning, local metric projection, candidate generation, and feature
  extraction are already done. Invalid GPS rows are dropped and unavailable
  motion features are explicitly masked. The feature columns have no missing
  values and need no categorical encoding. Fit scaling inside your training
  pipeline where your model needs it.
- Load IDs as strings and use only the listed feature columns as inputs.
  Observation diagnostics contain reference-derived fields for evaluation.
- The verified export has 1,103,062 training rows and 113,639 student-test
  rows. Load the full CSV programmatically because training exceeds the
  standard Excel worksheet row limit. No GPS observations were dropped from
  this archive. See the [dataset report](DATASET_REPORT.md) for coverage.

```python
import pandas as pd
from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

train = pd.read_csv("outputs/train/candidates.csv", dtype={"trip_id": str})
student_test = pd.read_csv("outputs/student_test/candidates.csv", dtype={"trip_id": str})
X_train = train[FEATURE_COLUMNS]
y_train = train[TARGET_COLUMN]
groups = train["trip_id"]
X_test = student_test[FEATURE_COLUMNS]
y_test = student_test[TARGET_COLUMN]
```

## Schema

- One row is one cleaned GPS observation × one candidate **directed** road arc.
- Unique row key: `trip_id`, `observation_id`, `candidate_arc_id`. IDs are strings
  for trips and zero-based original file row indices for observations and arcs.
  An arc ID is local to its trip. Group folds by `trip_id`, and group candidate
  scores by `(trip_id, observation_id)` for Top-1 and the decoder.
- Pass only `FEATURE_COLUMNS` as X, in the order below. All features are numeric
  and finite. `target` is binary: 1 for the candidate whose arc ID equals the
  aligned reference arc, 0 for other candidates. A candidate miss yields an
  all-zero group; report missing positives from `observations.candidate_miss`
  instead of dropping those groups from metrics. Expect at most one positive per
  observation. The reference route is aligned in order using minimum projected
  distance, so these labels are derived and can be ambiguous.
- `labeled=False` uses exactly the same features and candidate generation,
  omits `target` and reference diagnostics, and does not read the reference.
  When loading CSV in pandas, use `dtype={"trip_id": str}` to preserve IDs.

| Feature | Meaning |
|---|---|
| `distance_to_road_m` | Perpendicular or endpoint distance to finite candidate segment |
| `along_segment_fraction` | Position of nearest point from the start of the directed arc, in [0, 1] |
| `segment_length_m` | Projected segment length |
| `heading_error_deg` | Absolute circular difference between GPS step and directed arc bearings, 0–180 |
| `gps_step_distance_m` | Distance from previous valid observation, where available |
| `time_delta_s` | Seconds since previous valid observation, where available |
| `gps_speed_mps` | Step distance divided by elapsed time, where available |
| `has_motion_context` | 1 for a nonzero step with a consecutive source row and a valid interval of at most 30 seconds |
| `candidate_rank` | 1-based order by distance, with ties broken by arc ID |
| `from_node_in_degree` | Number of arcs entering the candidate's start node |
| `to_node_out_degree` | Number of arcs leaving the candidate's end node |

With no motion context the four motion values are zero, including heading error.
Candidate defaults are 50 m and at most 8 arcs. If the search radius is empty,
the closest arc is returned. Reference arcs are never injected into candidates.
Use the same settings in training and inference.
