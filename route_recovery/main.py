"""Development API: uvicorn route_recovery.main:app --host 0.0.0.0."""
import logging
import time
from pathlib import Path

import folium
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .hmm import RoadTrip, recover

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='PathSnap development route recovery')


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r'^\d{8}$')
    partition: str = Field(default='student_test', pattern=r'^(train|student_test)$')
    candidate_count: int = Field(default=5, ge=1, le=10)
    radius_m: float = Field(default=100, gt=0, le=500)


@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'distance_baseline_hmm', 'classifier_loaded': False}


@app.get('/trips')
def trips():
    return {part: sorted(p.name for p in (ROOT / part).iterdir() if p.is_dir()
                         and (p / (p.name + '.track')).is_file())
            if (ROOT / part).exists() else [] for part in ('train', 'student_test')}


@app.post('/match')
def match(request: MatchRequest):
    folder = ROOT / request.partition / request.trip_id
    if not (folder / (request.trip_id + '.track')).is_file():
        raise HTTPException(404, 'Trip not found')
    started = time.perf_counter()
    logging.getLogger('uvicorn.error').info('Starting route reconstruction: %s/%s', request.partition, request.trip_id)
    try:
        trip = RoadTrip(folder, request.trip_id)
        result = recover(trip, candidate_count=request.candidate_count, radius_m=request.radius_m)
    except (ValueError, IndexError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    result['elapsed_seconds'] = time.perf_counter() - started
    result['trip_id'] = request.trip_id
    logging.getLogger('uvicorn.error').info('Matched %s/%s: %s/%s in %.2fs', request.partition,
        request.trip_id, result['matched_observations'], result['observation_count'], result['elapsed_seconds'])
    return result


@app.post('/map', response_class=HTMLResponse)
def map_view(request: MatchRequest):
    result = match(request)
    view = folium.Map(location=result['gps'][0], zoom_start=14)
    gps = folium.FeatureGroup(name='GPS observations (blue)')
    for point in result['gps']:
        folium.CircleMarker(point, radius=2, color='blue', fill=True).add_to(gps)
    gps.add_to(view)
    route = folium.FeatureGroup(name='Predicted road sections (green)')
    for section in result['route_sections']:
        if len(section) > 1:
            folium.PolyLine(section, color='green', weight=5).add_to(route)
        else:
            folium.CircleMarker(section[0], color='green', radius=4).add_to(route)
    route.add_to(view)
    view.fit_bounds(result['gps'])
    folium.LayerControl().add_to(view)
    summary = f"<div style='position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:12px'>Distance baseline | Matched {result['matched_observations']}/{result['observation_count']} | Breaks {len(result['breaks'])} | {result['elapsed_seconds']:.2f}s</div>"
    view.get_root().html.add_child(folium.Element(summary))
    return view.get_root().render()


@app.get('/', response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')
