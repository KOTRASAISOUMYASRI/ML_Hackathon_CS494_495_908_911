"""Export feature tables and exploratory summaries, or serve the exploration page."""
import argparse
from html import escape
import json
from pathlib import Path

import folium
import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from data_loader import discover_trips, load_trip
from feature_engineering import build_feature_table
from preprocessing import clean_track, summarize_track
from road_network import RoadNetwork


def build_map(trip, track):
    center = trip.nodes[0, ::-1].tolist()
    result = folium.Map(location=center, zoom_start=14)
    roads = folium.FeatureGroup(name="Road network")
    folium.GeoJson({
        "type": "MultiLineString",
        "coordinates": trip.nodes[trip.arcs].tolist(),
    }, style_function=lambda _: {"color": "#64748b", "weight": 2}).add_to(roads)
    roads.add_to(result)
    gps = folium.FeatureGroup(name="Original GPS observations (valid coordinates)")
    coords_valid = trip.track.longitude.abs().le(180) & trip.track.latitude.abs().le(90)
    for row in trip.track.loc[coords_valid].itertuples():
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=3, color="#2563eb",
            tooltip=f"Observation {row.observation_id}",
        ).add_to(gps)
    gps.add_to(result)
    if trip.reference_arcs is not None:
        reference = folium.FeatureGroup(name="Development reference route")
        folium.GeoJson({
            "type": "MultiLineString",
            "coordinates": trip.nodes[trip.arcs[trip.reference_arcs]].tolist(),
        }, style_function=lambda _: {"color": "#15803d", "weight": 4}).add_to(reference)
        reference.add_to(result)
    coords = track[["latitude", "longitude"]].to_numpy()
    if len(coords):
        result.fit_bounds([coords.min(axis=0).tolist(), coords.max(axis=0).tolist()], max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(result)
    return result


def save_eda_plots(track, summary, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    trajectory, intervals, lengths, removed = figure.subplots(2, 2).flat
    trajectory.plot(track.longitude, track.latitude, marker=".", linewidth=0.6)
    trajectory.set(xlabel="Longitude", ylabel="Latitude", title="Representative trajectory")
    intervals.hist(track.timestamp.diff().dropna(), bins=20)
    intervals.set(xlabel="Seconds", ylabel="Steps", title="Representative sampling intervals")
    lengths.hist([row["trajectory_length_m"] for row in summary], bins=20)
    lengths.set(xlabel="Meters", ylabel="Trips", title="Trajectory lengths by trip")
    removed.hist([row["removed_observations"] for row in summary], bins=20)
    removed.set(xlabel="Removed rows", ylabel="Trips", title="GPS rows removed by trip")
    figure.savefig(destination / "eda_plots.png", dpi=150)


def summarize_trip(files, trip, track):
    road = RoadNetwork(trip.nodes, trip.arcs)
    paths = (files.track, files.nodes, files.arcs, files.route)
    projected = road.project(track[["longitude", "latitude"]].to_numpy())
    return {
        "trip_id": trip.trip_id,
        "files": {path.suffix: {"name": path.name, "bytes": path.stat().st_size}
                  for path in paths if path is not None},
        "road_nodes": len(trip.nodes),
        "road_arcs": len(trip.arcs),
        "zero_length_arcs": int((road.lengths == 0).sum()),
        "weak_components": nx.number_weakly_connected_components(road.graph),
        "isolated_nodes": nx.number_of_isolates(road.graph),
        "trajectory_length_m": float(np.linalg.norm(np.diff(projected, axis=0), axis=1).sum()),
        "reference_arc_count": len(trip.reference_arcs) if trip.reference_arcs is not None else None,
        **summarize_track(trip.track, track),
    }


def export_partition(partition_dir, destination, *, labeled=False):
    trips = discover_trips(partition_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, files in enumerate(trips):
        trip = load_trip(files, with_reference=labeled)
        result = build_feature_table(trip, labeled=labeled)
        for name, frame in (("candidates", result.candidates), ("observations", result.observations)):
            frame.to_csv(destination / f"{name}.csv", index=False,
                         mode="w" if index == 0 else "a", header=index == 0)
        info = summarize_trip(files, trip, result.observations)
        if labeled:
            info["candidate_misses"] = int(result.observations.candidate_miss.sum())
        summary.append(info)
        if index == 0:
            representative = result.observations
            build_map(trip, result.observations).save(str(destination / "sample_map.html"))
    (destination / "eda.json").write_text(
        json.dumps({"trip_count": len(trips), "trips": summary}, indent=2), encoding="utf-8"
    )
    pd.json_normalize(summary).to_csv(destination / "trip_summary.csv", index=False)
    save_eda_plots(representative, summary, destination)


def create_app(partition_dir, *, labeled=False):
    """Serve exploration for a training or student-test partition."""
    trips = discover_trips(partition_dir)
    app = FastAPI(title="PathSnap dataset exploration")

    @app.get("/", response_class=HTMLResponse)
    def index():
        links = "".join(
            f'<li><a href="/trips/{i}">{escape(files.trip_id)}</a></li>'
            for i, files in enumerate(trips)
        )
        return f"<h1>Dataset exploration</h1><p>{len(trips)} trips</p><ul>{links}</ul>"

    @app.get("/trips/{index}", response_class=HTMLResponse)
    def show_trip(index: int):
        if not 0 <= index < len(trips):
            raise HTTPException(404, "Unknown trip")
        files = trips[index]
        trip = load_trip(files, with_reference=labeled)
        track = clean_track(trip.track)
        stats = summarize_trip(files, trip, track)
        summary = escape(json.dumps(stats, indent=2))
        title = escape(trip.trip_id)
        map_document = escape(build_map(trip, track).get_root().render(), quote=True)
        return (
            f'<a href="/">All trips</a><h1>{title}</h1>'
            f'<iframe title="Trip map" style="width:100%;height:65vh" srcdoc="{map_document}"></iframe>'
            f"<h2>Trip summary</h2><pre>{summary}</pre>"
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", type=Path, help="Training or student-test folder")
    parser.add_argument("--output", type=Path, help="Directory for CSV, JSON and map exports")
    parser.add_argument("--serve", action="store_true", help="Serve the local dataset exploration UI")
    parser.add_argument("--labeled", action="store_true", help="Open permitted development references")
    args = parser.parse_args()
    if args.output:
        export_partition(args.partition, args.output, labeled=args.labeled)
    if args.serve:
        import uvicorn

        uvicorn.run(create_app(args.partition, labeled=args.labeled), host="127.0.0.1", port=8001)
    elif not args.output:
        parser.error("Specify --output or --serve")


if __name__ == "__main__":
    main()
