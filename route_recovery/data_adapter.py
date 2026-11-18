"""Bridge the shared, label-free Member 1 pipeline to route decoding."""
from pathlib import Path

import numpy as np

from data_loader import TripFiles, load_trip
from feature_engineering import build_feature_table
from road_network import RoadNetwork

from .hmm import Candidate, RoadTrip


class SharedRoadTrip(RoadTrip):
    """Reuse road-path routines with the official projection and feature rows."""

    def __init__(self, files, *, radius_m=50, max_candidates=8):
        trip = load_trip(files, with_reference=False)
        self.features = build_feature_table(
            trip, labeled=False, radius_m=radius_m, max_candidates=max_candidates)
        observations = self.features.observations
        if observations.empty:
            raise ValueError('No valid GPS observations remain after cleaning')
        self.observation_ids = observations.observation_id.astype(int).tolist()
        self.track = list(observations[['longitude', 'latitude', 'timestamp']].itertuples(index=False, name=None))
        self.nodes = trip.nodes.tolist()
        self.arcs = trip.arcs.tolist()
        self.network = RoadNetwork(trip.nodes, trip.arcs)
        self.lengths = self.network.lengths.tolist()
        self.adjacency = {}
        for arc, (u, v) in enumerate(self.arcs):
            self.adjacency.setdefault(u, []).append((v, self.lengths[arc], arc))
        self.candidate_rows = []
        groups = {int(key): group for key, group in self.features.candidates.groupby('observation_id', sort=False)}
        for observation_id in self.observation_ids:
            group = groups[observation_id]
            row = []
            for record in group.itertuples(index=False):
                arc = int(record.candidate_arc_id)
                fraction = float(record.along_segment_fraction)
                xy = self.network.starts[arc] + fraction * self.network.vectors[arc]
                row.append(Candidate(arc, fraction, float(record.distance_to_road_m), tuple(xy)))
            self.candidate_rows.append(row)

    def project(self, lon, lat):
        return tuple(self.network.project(np.array([[lon, lat]]))[0])

    def unproject(self, xy):
        from pyproj.enums import TransformDirection
        lon, lat = self.network.transformer.transform(*xy, direction=TransformDirection.INVERSE)
        return [float(lat), float(lon)]


def trip_files(folder, trip_id):
    base = Path(folder) / trip_id
    # Reference is intentionally absent, even when a .route exists on disk.
    return TripFiles(trip_id, base.with_suffix('.track'), base.with_suffix('.nodes'),
                     base.with_suffix('.arcs'), None)
