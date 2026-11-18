"""Build the shared candidate table for model training and inference."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing import clean_track
from road_network import RoadNetwork

ID_COLUMNS = ["trip_id", "observation_id", "candidate_arc_id"]
FEATURE_COLUMNS = [
    "distance_to_road_m", "along_segment_fraction", "segment_length_m",
    "heading_error_deg", "gps_step_distance_m", "time_delta_s", "gps_speed_mps",
    "has_motion_context", "candidate_rank", "from_node_in_degree", "to_node_out_degree",
]
TARGET_COLUMN = "target"


@dataclass(frozen=True)
class FeatureResult:
    candidates: pd.DataFrame
    observations: pd.DataFrame


def _align_reference(points, network, route):
    """Minimize squared distance over nondecreasing route positions."""
    if not len(points):
        return np.empty(0, dtype=np.int64)
    previous = network.segment_distances(points[0], route)[0] ** 2
    parents = np.empty((len(points), len(route)), dtype=np.int32)
    positions = np.arange(len(route))
    for index in range(1, len(points)):
        prefix_min = np.minimum.accumulate(previous)
        # Keep the earliest position when costs tie to make alignment reproducible.
        new_min = np.r_[True, previous[1:] < prefix_min[:-1]]
        best = np.maximum.accumulate(np.where(new_min, positions, 0))
        parents[index] = best
        previous = previous[best] + network.segment_distances(points[index], route)[0] ** 2
    position = int(previous.argmin())
    aligned = np.empty(len(points), dtype=np.int64)
    for index in range(len(points) - 1, -1, -1):
        aligned[index] = route[position]
        if index:
            position = parents[index, position]
    return aligned


def build_feature_table(trip, *, labeled=False, radius_m=50.0, max_candidates=8, gap_seconds=30.0):
    """Generate identical candidates for training and inference, then attach labels."""
    if (not np.isfinite(radius_m) or radius_m <= 0
            or not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1):
        raise ValueError("Expected a positive radius and at least one candidate")
    if not np.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ValueError("Expected a positive gap threshold")
    if labeled and (trip.reference_arcs is None or not len(trip.reference_arcs)):
        raise ValueError(f"No reference route loaded for {trip.trip_id}")

    track = clean_track(trip.track)
    network = RoadNetwork(trip.nodes, trip.arcs)
    points = network.project(track[["longitude", "latitude"]].to_numpy())
    observations = track.copy()
    observations.insert(0, "trip_id", trip.trip_id)
    ids = track.observation_id.to_numpy()
    steps = np.diff(points, axis=0, prepend=points[:1])
    step_distances = np.linalg.norm(steps, axis=1)
    deltas = track.timestamp.diff().to_numpy()
    motion = (
        (np.diff(ids, prepend=ids[:1]) == 1)
        & (deltas > 0) & (deltas <= gap_seconds) & (step_distances > 0)
    )

    rows = []
    counts = []
    for index, point in enumerate(points):
        candidates, distances, fractions = network.candidates(point, radius_m, max_candidates)
        counts.append(len(candidates))
        step = elapsed = speed = 0.0
        heading_errors = np.zeros(len(candidates))
        if motion[index]:
            step = step_distances[index]
            elapsed = deltas[index]
            speed = step / elapsed
            heading = np.degrees(np.arctan2(*steps[index]))
            heading_errors = np.abs((heading - network.bearings[candidates] + 180) % 360 - 180)
        for rank, (arc, distance, fraction, error) in enumerate(
            zip(candidates, distances, fractions, heading_errors), 1
        ):
            start, end = trip.arcs[arc]
            rows.append((
                trip.trip_id, ids[index], arc, distance, fraction, network.lengths[arc],
                error, step, elapsed, speed, int(motion[index]), rank,
                network.graph.in_degree(start), network.graph.out_degree(end),
            ))

    table = pd.DataFrame(rows, columns=ID_COLUMNS + FEATURE_COLUMNS)
    for column in ID_COLUMNS[1:] + FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column])
    observations["candidate_count"] = np.asarray(counts, dtype=np.int64)
    if labeled:
        aligned = _align_reference(points, network, trip.reference_arcs)
        reference = pd.Series(aligned, index=ids)
        table[TARGET_COLUMN] = table.candidate_arc_id.eq(
            table.observation_id.map(reference)
        ).astype("int8")
        positives = table.loc[table.target.eq(1), "observation_id"]
        observations["reference_arc_id"] = aligned
        observations["candidate_miss"] = ~observations.observation_id.isin(positives)
    return FeatureResult(table, observations)
