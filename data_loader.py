"""Read PathSnap trip files without opening evaluator reference routes."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripFiles:
    trip_id: str
    track: Path
    nodes: Path
    arcs: Path
    route: Path | None


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: pd.DataFrame
    nodes: np.ndarray
    arcs: np.ndarray
    reference_arcs: np.ndarray | None = None


def discover_trips(partition_dir):
    """Find same-stem trip files. IDs are relative paths without extensions."""
    root = Path(partition_dir)
    if not root.is_dir():
        raise ValueError(f"Partition directory does not exist: {root}")
    tracks = sorted(root.rglob("*.track"))
    if not tracks:
        raise ValueError(f"No .track files in {root}")
    trips = []
    for track in tracks:
        nodes = track.with_suffix(".nodes")
        arcs = track.with_suffix(".arcs")
        route = track.with_suffix(".route")
        for required in (nodes, arcs):
            if not required.is_file():
                raise ValueError(f"Missing trip file: {required}")
        trips.append(TripFiles(
            track.relative_to(root).with_suffix("").as_posix(),
            track, nodes, arcs, route if route.is_file() else None,
        ))
    return trips


def _read_rows(path, columns, dtype):
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            values = line.split()
            if len(values) != columns:
                raise ValueError(f"{path}:{line_number}: expected {columns} fields")
            try:
                rows.append([dtype(value) for value in values])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from error
    if not rows:
        raise ValueError(f"Empty file: {path}")
    return np.asarray(rows, dtype=dtype)


def _read_track(path):
    """Keep source line indices even when a GPS row is malformed."""
    def parse(value):
        try:
            return float(value)
        except ValueError:
            return np.nan

    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            values = line.split()
            records.append([parse(value) for value in values] if len(values) == 3
                           else [np.nan] * 3)
    track = pd.DataFrame(
        records, columns=["longitude", "latitude", "timestamp"], dtype=float
    )
    track.insert(0, "observation_id", np.arange(len(track), dtype=np.int64))
    return track


def load_trip(files, *, with_reference=False):
    nodes = _read_rows(files.nodes, 2, float)
    arcs = _read_rows(files.arcs, 2, int)
    if not np.isfinite(nodes).all() or np.any(np.abs(nodes) > [180, 90]):
        raise ValueError(f"Invalid road coordinates in {files.nodes}")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid node ID in {files.arcs}")
    reference = None
    if with_reference:
        if files.route is None:
            raise ValueError(f"Missing .route for {files.trip_id}")
        reference = _read_rows(files.route, 1, int).ravel()
        if np.any(reference < 0) or np.any(reference >= len(arcs)):
            raise ValueError(f"Invalid arc ID in {files.route}")
    return Trip(files.trip_id, _read_track(files.track), nodes, arcs, reference)
