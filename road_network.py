"""Directed roads, trip-local metric projection, and candidate queries."""
import networkx as nx
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


class RoadNetwork:
    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs
        longitude, latitude = nodes[0]
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m"
        )
        self.transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        xy = self.project(nodes)
        self.starts = xy[arcs[:, 0]]
        self.vectors = xy[arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.bearings = np.degrees(np.arctan2(self.vectors[:, 0], self.vectors[:, 1]))
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(range(len(nodes)))
        self.graph.add_edges_from(
            (int(a), int(b), i, {"arc_id": i, "length_m": float(self.lengths[i])})
            for i, (a, b) in enumerate(arcs)
        )
        self.search_ids = np.flatnonzero(self.lengths > 0)
        if not len(self.search_ids):
            raise ValueError("Road network has no nonzero-length arcs")
        self.max_half_length = float(self.lengths.max() / 2)
        self.index = cKDTree(self.starts[self.search_ids] + self.vectors[self.search_ids] / 2)

    def project(self, lon_lat):
        x, y = self.transformer.transform(lon_lat[:, 0], lon_lat[:, 1], errcheck=True)
        return np.column_stack((x, y))

    def segment_distances(self, point, ids):
        offset = point - self.starts[ids]
        vectors = self.vectors[ids]
        squared = self.lengths[ids] ** 2
        fraction = np.divide(
            np.einsum("ij,ij->i", offset, vectors), squared,
            out=np.zeros(len(ids)), where=squared > 0,
        ).clip(0, 1)
        distances = np.linalg.norm(offset - fraction[:, None] * vectors, axis=1)
        return distances, fraction

    def candidates(self, point, radius_m, max_candidates):
        """Return exact radius-filtered segments or the global nearest fallback."""
        # A segment within radius must have its midpoint within radius + half length.
        ids = self.search_ids[self.index.query_ball_point(point, radius_m + self.max_half_length)]
        distances, fractions = self.segment_distances(point, ids)
        keep = distances <= radius_m
        if keep.any():
            ids, distances, fractions = ids[keep], distances[keep], fractions[keep]
            order = np.lexsort((ids, distances))[:max_candidates]
        else:
            _, closest_midpoint = self.index.query(point)
            nearest_id = self.search_ids[[closest_midpoint]]
            upper_bound = self.segment_distances(point, nearest_id)[0][0]
            bound = np.nextafter(upper_bound + self.max_half_length, np.inf)
            ids = self.search_ids[self.index.query_ball_point(point, bound)]
            distances, fractions = self.segment_distances(point, ids)
            order = np.lexsort((ids, distances))[:1]
        return ids[order], distances[order], fractions[order]
