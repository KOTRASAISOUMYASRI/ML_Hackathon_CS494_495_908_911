# Dataset understanding and preprocessing

The development workflow inventories same-stem .track, .nodes, .arcs, and
.route files by partition. GPS observations retain their original row IDs.
Road arcs retain their directed endpoints and positional arc IDs, including
parallel arcs. Weak components and isolates describe network connectivity.

Rows with missing, malformed, nonfinite, or out-of-range measurements are
removed. Within each trip the pipeline retains the original file order and
drops repeated or backward timestamps relative to the retained timeline.
It records irregular intervals and gaps over 30 seconds. Motion features are
masked across those gaps and across missing source rows.

For each trip, a WGS84 azimuthal equidistant projection centered on its first
road node gives local metric coordinates. Verify each trip's extent is suitable
for this local approximation. Finite-segment distances rank nearby candidates,
with a nearest-arc fallback where the search radius is empty. Directed arc
length, position, heading agreement, time and speed, candidate rank, and
endpoint degrees capture local geometry, motion, and connectivity. Reference
routes are aligned in traversal order by minimizing squared segment distances.
These derived labels can be uncertain around parallel roads or junctions.
Measure candidate misses before interpreting classifier performance.

Run the exploration export separately for training and student-testing. Use
the resulting counts, missingness, interval and trajectory-length summaries,
coordinate bounds and graph statistics to write the observed EDA findings.
Export a representative Folium view with the GPS, road and available
development reference layers. The final team map also needs the model's
predicted route. Only insert metrics and maps from the actual dataset and
identify which partition produced each result.
