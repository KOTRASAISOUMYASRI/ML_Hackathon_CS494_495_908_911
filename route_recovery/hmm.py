"""Development distance-emission HMM and replaceable candidate-score interface.

Reference route files are deliberately never read by this module.
"""

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import cos, hypot, isfinite, radians
from pathlib import Path

from .viterbi import decode


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    fraction: float
    distance_m: float
    xy: tuple


class RoadTrip:
    """Local equirectangular projection in metres, centred on the trip graph.

    Suitable as a local baseline; assess distortion before using large extents.
    Node and arc IDs are zero-based file row positions.
    """

    def __init__(self, folder, trip_id):
        base = Path(folder) / trip_id
        self.nodes = [tuple(map(float, r.split())) for r in base.with_suffix('.nodes').read_text().splitlines() if r.strip()]
        self.arcs = [tuple(map(int, r.split())) for r in base.with_suffix('.arcs').read_text().splitlines() if r.strip()]
        self.track = [tuple(map(float, r.split())) for r in base.with_suffix('.track').read_text().splitlines() if r.strip()]
        if not self.nodes or not self.track or any(len(r) != 2 for r in self.nodes + self.arcs) or any(len(r) != 3 for r in self.track):
            raise ValueError('Empty or malformed trip')
        if any(not all(isfinite(v) for v in r) or abs(r[0]) > 180 or abs(r[1]) > 90 for r in self.nodes + self.track):
            raise ValueError('Invalid coordinates or timestamps')
        if any(b[2] <= a[2] for a, b in zip(self.track, self.track[1:])):
            raise ValueError('Timestamps must be strictly increasing')
        self.lon0, self.lat0 = self.nodes[0]
        self.xscale = 6371000 * cos(radians(self.lat0))
        self.xy = [self.project(*p) for p in self.nodes]
        self.lengths, self.adjacency = [], {}
        for arc_id, (u, v) in enumerate(self.arcs):
            if not (0 <= u < len(self.nodes) and 0 <= v < len(self.nodes)):
                raise ValueError('Arc refers to missing node')
            length = hypot(self.xy[v][0] - self.xy[u][0], self.xy[v][1] - self.xy[u][1])
            self.lengths.append(length)
            self.adjacency.setdefault(u, []).append((v, length, arc_id))

    def project(self, lon, lat):
        return radians(lon - self.lon0) * self.xscale, radians(lat - self.lat0) * 6371000

    def unproject(self, xy):
        from math import degrees
        return [self.lat0 + degrees(xy[1] / 6371000), self.lon0 + degrees(xy[0] / self.xscale)]

    def candidates(self, observation, count=5, radius=100):
        p = self.project(*observation[:2])
        found = []
        for arc_id, (u, v) in enumerate(self.arcs):
            a, b = self.xy[u], self.xy[v]
            if (p[0] < min(a[0], b[0]) - radius or p[0] > max(a[0], b[0]) + radius
                    or p[1] < min(a[1], b[1]) - radius or p[1] > max(a[1], b[1]) + radius):
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            denominator = dx * dx + dy * dy
            fraction = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denominator)) if denominator else 0
            q = (a[0] + fraction * dx, a[1] + fraction * dy)
            distance = hypot(p[0] - q[0], p[1] - q[1])
            if distance <= radius:
                found.append(Candidate(arc_id, fraction, distance, q))
        return sorted(found, key=lambda c: (c.distance_m, c.arc_id))[:count]

    @lru_cache(maxsize=4096)
    def shortest(self, source, target, cutoff=float('inf')):
        queue, distances, parents = [(0, source)], {source: 0}, {}
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if distance > cutoff:
                break
            if node == target:
                path = []
                while node != source:
                    node, arc = parents[node]
                    path.append(arc)
                return distance, tuple(reversed(path))
            for neighbor, length, arc in self.adjacency.get(node, []):
                value = distance + length
                if value <= cutoff and value < distances.get(neighbor, float('inf')):
                    distances[neighbor] = value
                    parents[neighbor] = (node, arc)
                    heappush(queue, (value, neighbor))
        return float('inf'), ()

    def connect(self, a, b, max_distance=float('inf')):
        if a.arc_id == b.arc_id and b.fraction >= a.fraction:
            return (b.fraction - a.fraction) * self.lengths[a.arc_id], ()
        partial = (1 - a.fraction) * self.lengths[a.arc_id] + b.fraction * self.lengths[b.arc_id]
        if partial > max_distance:
            return float('inf'), ()
        distance, path = self.shortest(self.arcs[a.arc_id][1], self.arcs[b.arc_id][0], max_distance - partial)
        return partial + distance, path


def recover(trip, *, candidate_count=8, radius_m=50, sigma_m=20, beta_m=30,
            max_gap_s=30, max_speed_mps=60, scorer=None, diagnose=False):
    """scorer(observation, candidates) returns aligned natural-log scores.

    Without scorer this is a distance baseline, not the required ML model.
    Breaks restart decoding explicitly; no line is drawn across a break.
    """
    if any(not isfinite(x) or x <= 0 for x in (radius_m, sigma_m, beta_m, max_gap_s, max_speed_mps)) or candidate_count < 1:
        raise ValueError('Configuration values must be positive and finite')
    candidates = (trip.candidate_rows if hasattr(trip, 'candidate_rows') else
                  [trip.candidates(o, candidate_count, radius_m) for o in trip.track])
    emissions = []
    for observation, row in zip(trip.track, candidates):
        values = list(scorer(observation, row)) if scorer is not None and row else [-0.5 * (c.distance_m / sigma_m) ** 2 for c in row]
        if len(values) != len(row) or any(not isfinite(v) and v != float('-inf') for v in values):
            raise ValueError('Scorer must return one valid log score per candidate')
        emissions.append(values)
    selected, sections, breaks = [None] * len(candidates), [], []
    start, active, matrices, forward = None, [], [], []

    def flush():
        if not active:
            return
        indices, _ = decode(active, matrices)
        section = []
        for offset, index in enumerate(indices):
            t = start + offset
            selected[t] = candidates[t][index]
            section.append(t)
        sections.append(section)

    for t, row in enumerate(candidates):
        if not row or not any(isfinite(v) for v in emissions[t]):
            flush()
            active, matrices, start, forward = [], [], None, []
            breaks.append({'observation': t, 'reason': 'no_usable_candidates'})
            continue
        if not active:
            start, active, forward = t, [emissions[t]], list(emissions[t])
            continue
        dt = trip.track[t][2] - trip.track[t - 1][2]
        ids = getattr(trip, 'observation_ids', None)
        if ids is not None and ids[t] != ids[t - 1] + 1:
            flush()
            breaks.append({'observation': t, 'reason': 'removed_observations'})
            start, active, matrices, forward = t, [emissions[t]], [], list(emissions[t])
            continue
        if dt > max_gap_s:
            flush()
            breaks.append({'observation': t, 'reason': 'timestamp_gap'})
            start, active, matrices, forward = t, [emissions[t]], [], list(emissions[t])
            continue
        observed = hypot(*(b - a for a, b in zip(trip.project(*trip.track[t - 1][:2]), trip.project(*trip.track[t][:2]))))
        matrix = [[(-abs(distance - observed) / beta_m if isfinite(distance) and distance <= max_speed_mps * dt else float('-inf'))
                   for distance in [trip.connect(a, b, max_speed_mps * dt)[0] for b in row]] for a in candidates[t - 1]]
        reason = 'timestamp_gap' if dt > max_gap_s else None
        next_scores = [emissions[t][j] + max(forward[i] + matrix[i][j] for i in range(len(forward))) for j in range(len(row))]
        if reason is None and not any(isfinite(v) for v in next_scores):
            reason = 'unreachable_transition'
        if reason:
            flush()
            detail = {'observation': t, 'reason': reason}
            if diagnose:
                eligible = [a for i, a in enumerate(candidates[t - 1]) if isfinite(forward[i])]
                targets = [b for j, b in enumerate(row) if isfinite(emissions[t][j])]
                minimum = minimum_connection(trip, eligible, targets)
                detail.update(cause='speed_limit' if isfinite(minimum) else 'no_directed_path',
                              minimum_network_m=minimum if isfinite(minimum) else None,
                              allowed_network_m=max_speed_mps * dt,
                              previous_candidate_count=len(eligible), candidate_count=len(targets),
                              nearest_candidate_m=min(c.distance_m for c in row))
                if not isfinite(minimum):
                    import networkx as nx
                    graph = nx.Graph()
                    graph.add_edges_from(trip.arcs)
                    components = {node: index for index, nodes in enumerate(nx.connected_components(graph)) for node in nodes}
                    connected = any(components.get(trip.arcs[a.arc_id][1]) == components.get(trip.arcs[b.arc_id][0]) for a in eligible for b in targets)
                    detail['cause'] = 'road_direction' if connected else 'disconnected_components'
            breaks.append(detail)
            start, active, matrices, forward = t, [emissions[t]], [], list(emissions[t])
        else:
            active.append(emissions[t])
            matrices.append(matrix)
            forward = next_scores
    flush()
    geometry = []
    for section in sections:
        points = [trip.unproject(selected[section[0]].xy)]
        for previous, current in zip(section, section[1:]):
            a, b = selected[previous], selected[current]
            _, middle = trip.connect(a, b)
            if a.arc_id != b.arc_id or b.fraction < a.fraction:
                points.append(list(reversed(trip.nodes[trip.arcs[a.arc_id][1]])))
                points.extend(list(reversed(trip.nodes[trip.arcs[arc][1]])) for arc in middle)
            points.append(trip.unproject(b.xy))
        geometry.append(points)
    return {'mode': 'classifier_hmm' if scorer else 'distance_baseline_hmm',
            'observation_ids': getattr(trip, 'observation_ids', list(range(len(selected)))),
            'arc_ids': [c.arc_id if c else None for c in selected],
            'independent_arc_ids': [row[max(range(len(row)), key=lambda j: emissions[t][j])].arc_id if row and any(isfinite(v) for v in emissions[t]) else None for t, row in enumerate(candidates)],
            'route_sections': geometry, 'breaks': breaks,
            'section_observation_indices': sections,
            'gps': [[o[1], o[0]] for o in trip.track],
            'matched_observations': sum(c is not None for c in selected),
            'observation_count': len(selected)}


def minimum_connection(trip, previous, following):
    """Exact unconstrained minimum directed travel distance between candidate sets.

    One multi-source Dijkstra with partial first/last arcs, for diagnostics only.
    """
    best = min(((b.fraction - a.fraction) * trip.lengths[a.arc_id]
                for a in previous for b in following
                if a.arc_id == b.arc_id and b.fraction >= a.fraction), default=float('inf'))
    distances, queue = {}, []
    for a in previous:
        node = trip.arcs[a.arc_id][1]
        value = (1 - a.fraction) * trip.lengths[a.arc_id]
        if value < distances.get(node, float('inf')):
            distances[node] = value
            heappush(queue, (value, node))
    endpoints = {}
    for b in following:
        node = trip.arcs[b.arc_id][0]
        endpoints[node] = min(endpoints.get(node, float('inf')), b.fraction * trip.lengths[b.arc_id])
    while queue:
        distance, node = heappop(queue)
        if distance != distances[node]:
            continue
        if distance >= best:
            break
        if node in endpoints:
            best = min(best, distance + endpoints[node])
        for neighbor, length, _ in trip.adjacency.get(node, []):
            value = distance + length
            if value < best and value < distances.get(neighbor, float('inf')):
                distances[neighbor] = value
                heappush(queue, (value, neighbor))
    return best
