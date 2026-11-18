# Development route recovery

## Shared data integration

The API now uses Member 1's `load_trip(with_reference=False)` and
`build_feature_table(labeled=False)` through `data_adapter.py`. Candidate lists,
order, distances and fractions come directly from the training feature table.
Defaults match the feature contract: radius 50 m, up to 8 directed arcs, and
nearest-arc fallback. Projection is the shared local azimuthal equidistant CRS.
Original observation IDs are returned, and removed rows split decoding sections.
The old standalone RoadTrip remains only for isolated decoder tests.

Set the dataset parent directory before starting, if it is outside the repo:

```powershell
$env:PATHSNAP_DATA_ROOT = 'C:/path/to/Student folder'
python -m uvicorn route_recovery.main:app --host 0.0.0.0 --port 8000
```

Trip IDs now follow the shared inventory (for example `00000052/00000052`).
Clients must use IDs returned by `/trips`; arbitrary paths are never accepted.
No reference route is loaded. Classifier integration is still pending.

## Student-test execution check

Run `python -m route_recovery.check_trips student_test` to reproduce the
execution check. Measured results are in
[`../reports/student_test_execution.json`](../reports/student_test_execution.json).
All ten supplied student-test trips completed with valid output shapes. Some
have many unreachable-transition breaks (97 for trip 00000040); these are
explicit route discontinuities, not successful continuity or accuracy results.
Investigate candidate coverage and decoder assumptions with the model owners
before drawing conclusions or selecting decoder parameters.

Distance-based emissions are a baseline, not the selected trained classifier.
This module never reads reference route files and reports no accuracy metrics.

From the repository root, with requirements installed:

```sh
python -m unittest discover -s tests -v
uvicorn route_recovery.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. Laptop 2 can open http://LAPTOP_1_LAN_IP:8000 on
the same network. Allow port 8000 if the firewall prompts. API logs appear on
Laptop 1. Folium assets and map tiles need internet access. `/docs` has API docs.
For separate Laptop 2 hosting, run
`python -m uvicorn route_recovery.ui_server:app --port 8080` on Laptop 2 and
enter Laptop 1's API address in the page. See [deployment steps](DEPLOYMENT.md)
for allowed origins, network configuration and physical-device verification.
The UI shows runtime, matched/unmatched counts, road sections, break details
and map layers. The `/visualize` endpoint returns the results and map together.
Diagnostic mode reports why transitions failed without changing predictions;
see [measured findings](../reports/ROUTE_BREAK_FINDINGS.md).

## Algorithm and integration

Viterbi accepts natural-log emission scores and previous-by-next transition
matrices. Negative infinity means impossible. Initial priors are uniform up to
a constant. Ties choose the first candidate. Returned indices refer to each
observation's own candidate list, preserving arc correspondence.

`recover(trip, scorer=callback)` accepts a callback `(observation, candidates)`
returning aligned log scores. Candidate fields: arc_id, fraction along directed
arc, distance_m and projected xy. Agree on feature ordering and probability
conversion with classifier owners before connecting their artifacts.

Transitions penalise network-versus-observed distance mismatch. Directed
Dijkstra includes partial end arcs. Speed and timestamp gap thresholds are
configurable. No candidates or impossible transitions produce explicit breaks;
missing matches remain null and geometry never bridges a break.

The API uses the shared indexed candidate queries. Dijkstra uses a bounded
cache. Benchmark large trips before demonstration. Decoder scoring defaults are
engineering starting points, not optimised parameters.

## Remaining work

- Integrate selected classifier emissions using the shared feature rows.
- Validate reference alignment with the data/evaluation owners; implement
  server-only evaluation and classifier-only versus HMM comparisons.
- Add reserved evaluator-trip configuration without exposing ground truth.
- Verify the implemented separate UI deployment on two physical laptops.
- Record actual dataset metrics, timings, maps and API evidence for the report.

Synthetic tests verify behaviour; they are not experimental dataset results.
