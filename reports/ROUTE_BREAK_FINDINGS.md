# Development route-break investigation

Source: [measured per-trip diagnostics](route_break_diagnostics.json), using
the shared 50 m / eight-candidate defaults and distance-based HMM emissions.
No reference route was read; this is not an accuracy evaluation.

All ten trips completed. The 334 explicit breaks consist of:

| Diagnostic | Count | Interpretation |
|---|---:|---|
| Speed-distance limit | 329 | No finite-score continuation fits the configured 60 m/s limit; a longer directed path exists |
| Disconnected components | 2 | Surviving previous and current candidates are in different weak road components |
| Road direction | 1 | Weak connectivity exists, but no directed continuation exists |
| Timestamp gap | 2 | Observation interval exceeds the 30-second threshold |

No observations required the shared nearest-road fallback outside 50 metres.
Trip 00000040 has 97 speed-limit breaks. Some required distances are small
threshold exceedances; others require long detours. These are diagnostics of
the current candidate sequence and graph, not evidence the vehicle actually
travelled at those speeds. Directed projection jitter, omitted alternatives and
accumulated decoder reachability can influence them.

The investigation does not justify blindly raising the speed limit, reversing
one-way roads, injecting reference candidates or drawing straight connectors.
The default parameters and prediction sequence were preserved. Next, compare
classifier emissions and validated reference alignment on permitted development
data before deciding whether to change the observation or transition model.

Tests cover backward movement with/without a directed return cycle, disconnected
roads, missing candidates, nearest-road fallback and intermediate route geometry.
Map breaks remain distinct sections and are marked in orange.
