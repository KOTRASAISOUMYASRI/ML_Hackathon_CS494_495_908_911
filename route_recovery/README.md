# Development route recovery

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
The development UI is served by Laptop 1; separate Laptop 2 hosting remains.

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

The loader and exhaustive candidate scan are replaceable adapters for the data
owner's modules. Dijkstra uses a bounded cache. Benchmark large trips before
demonstration. Local equirectangular projection is a baseline requiring distortion
checks for the dataset extent. Defaults are engineering starting points, not
optimised parameters.

## Remaining work

- Integrate official candidates/features and selected classifier emissions.
- Validate reference alignment with the data/evaluation owners; implement
  server-only evaluation and classifier-only versus HMM comparisons.
- Add reserved evaluator-trip configuration without exposing ground truth.
- Verify separate UI deployment on two physical laptops.
- Record actual dataset metrics, timings, maps and API evidence for the report.

Synthetic tests verify behaviour; they are not experimental dataset results.
