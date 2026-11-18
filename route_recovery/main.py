"""Development API: uvicorn route_recovery.main:app --host 0.0.0.0."""
import logging
import time
import os
from pathlib import Path

import folium
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .hmm import recover
from .data_adapter import SharedRoadTrip
from data_loader import discover_trips
from .classifier import model_paths, score_candidates
from .evaluation import evaluate_reference
from .selection import SELECTED_MODEL, selected_comparison

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('PATHSNAP_DATA_ROOT', str(ROOT)))
EVALUATOR_INPUT = os.environ.get('PATHSNAP_EVALUATOR_INPUT', str(ROOT / 'evaluator_test'))
EVALUATOR_REFERENCE = os.environ.get('PATHSNAP_EVALUATOR_REFERENCE')
app = FastAPI(title='PathSnap development route recovery')
app.add_middleware(CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get('PATHSNAP_UI_ORIGINS', 'http://localhost:8080,http://127.0.0.1:8080').split(',') if origin.strip()],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


class MatchRequest(BaseModel):
    trip_id: str = Field(min_length=1)
    partition: str = Field(default='student_test', pattern=r'^(train|student_test|evaluator)$')
    candidate_count: int = Field(default=8, ge=1, le=10)
    radius_m: float = Field(default=50, gt=0, le=500)
    diagnose: bool = False
    model: str = Field(default=SELECTED_MODEL, min_length=1)
    evaluate: bool = False


@app.get('/health')
def health():
    return {'status': 'ok', 'models': ['distance', *model_paths()],
            'default_model': SELECTED_MODEL, 'final_model_selected': True,
            'default_model_available': SELECTED_MODEL in model_paths()}


@app.get('/selected-model')
def selected_model():
    return selected_comparison()


@app.get('/trips')
def trips():
    return {part: list(inventory(part)) for part in ('train', 'student_test', 'evaluator')}


def inventory(partition):
    if partition == 'evaluator' and not EVALUATOR_INPUT:
        return {}
    folder = Path(EVALUATOR_INPUT) if partition == 'evaluator' else DATA_ROOT / partition
    if not folder.is_dir():
        return {}
    try:
        return {files.trip_id: files for files in discover_trips(folder)}
    except ValueError as exc:
        logging.getLogger('uvicorn.error').warning('Dataset inventory: %s', exc)
        return {}


@app.post('/match')
def match(request: MatchRequest):
    files = inventory(request.partition).get(request.trip_id)
    if files is None:
        raise HTTPException(404, 'Trip not found')
    started = time.perf_counter()
    logging.getLogger('uvicorn.error').info('Starting route reconstruction: %s/%s model=%s', request.partition, request.trip_id, request.model)
    try:
        trip = SharedRoadTrip(files, radius_m=request.radius_m, max_candidates=request.candidate_count)
        if request.model != 'distance' and (request.radius_m != 50 or request.candidate_count != 8):
            raise ValueError('Saved classifiers require training settings: 50 metres, 8 candidates')
        scoring_start = time.perf_counter()
        scorer, probabilities = ((None, None) if request.model == 'distance'
                                 else score_candidates(trip, request.model))
        scoring_seconds = time.perf_counter() - scoring_start
        decoding_start = time.perf_counter()
        result = recover(trip, candidate_count=request.candidate_count, radius_m=request.radius_m,
                         diagnose=request.diagnose, scorer=scorer)
        result['decoder_seconds'] = time.perf_counter() - decoding_start
        result['classifier_seconds'] = scoring_seconds
    except (ValueError, IndexError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    result['elapsed_seconds'] = time.perf_counter() - started
    result['trip_id'] = request.trip_id
    result['model'] = request.model
    result['final_model_selected'] = True
    result['is_selected_model'] = request.model == SELECTED_MODEL
    if request.evaluate:
        reference_files = files
        if request.partition == 'evaluator':
            if not EVALUATOR_REFERENCE:
                raise HTTPException(409, 'Server-side evaluator reference directory is not configured')
            from data_loader import TripFiles
            route = Path(EVALUATOR_REFERENCE) / (files.trip_id + '.route')
            reference_files = TripFiles(files.trip_id, files.track, files.nodes, files.arcs, route)
        evaluation_start = time.perf_counter()
        try:
            result['evaluation'] = evaluate_reference(trip, result, reference_files, probabilities)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, 'Reference evaluation unavailable: ' + str(exc)) from exc
        result['evaluation_seconds'] = time.perf_counter() - evaluation_start
    result['unmatched_observations'] = result['observation_count'] - result['matched_observations']
    result['section_count'] = len(result['route_sections'])
    logging.getLogger('uvicorn.error').info('Matched %s/%s: %s/%s in %.2fs', request.partition,
        request.trip_id, result['matched_observations'], result['observation_count'], result['elapsed_seconds'])
    return result


@app.post('/map', response_class=HTMLResponse)
def map_view(request: MatchRequest):
    result = match(request)
    view = render_map(result, request)
    return view.get_root().render()


@app.post('/visualize')
def visualize(request: MatchRequest):
    result = match(request)
    view = render_map(result, request)
    return {'result': result, 'map_html': view.get_root().render()}


def render_map(result, request):
    view = folium.Map(location=result['gps'][0], zoom_start=14)
    gps = folium.FeatureGroup(name='GPS observations (blue)')
    for point in result['gps']:
        folium.CircleMarker(point, radius=2, color='blue', fill=True).add_to(gps)
    gps.add_to(view)
    files = inventory(request.partition).get(request.trip_id)
    from data_loader import load_trip
    trip = load_trip(files, with_reference=False)
    roads = folium.FeatureGroup(name='Road network (grey)', show=False)
    folium.GeoJson({'type': 'Feature', 'properties': {}, 'geometry': {
        'type': 'MultiLineString', 'coordinates': [[trip.nodes[u].tolist(), trip.nodes[v].tolist()] for u, v in trip.arcs]}},
        style_function=lambda _: {'color': '#718096', 'weight': 2, 'opacity': .5}).add_to(roads)
    roads.add_to(view)
    independent = folium.FeatureGroup(name='Independent selected segments (purple)', show=False)
    selected_ids = sorted({arc for arc in result['independent_arc_ids'] if arc is not None})
    folium.GeoJson({'type':'Feature', 'properties':{}, 'geometry':{
        'type':'MultiLineString', 'coordinates':[[trip.nodes[trip.arcs[arc][0]].tolist(),
                                                trip.nodes[trip.arcs[arc][1]].tolist()] for arc in selected_ids]}},
        style_function=lambda _: {'color':'purple','weight':3,'opacity':.65}).add_to(independent)
    independent.add_to(view)
    gaps = folium.FeatureGroup(name='Route breaks (orange)')
    for gap in result['breaks']:
        folium.CircleMarker(result['gps'][gap['observation']], radius=6, color='orange',
                            tooltip=gap['reason']).add_to(gaps)
    gaps.add_to(view)
    route = folium.FeatureGroup(name='Predicted road sections (green)')
    for section in result['route_sections']:
        if len(section) > 1:
            folium.PolyLine(section, color='green', weight=5).add_to(route)
        else:
            folium.CircleMarker(section[0], color='green', radius=4).add_to(route)
    route.add_to(view)
    view.fit_bounds(result['gps'])
    folium.LayerControl().add_to(view)
    summary = f"<div style='position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:12px'>{result['mode']} | Matched {result['matched_observations']}/{result['observation_count']} | Breaks {len(result['breaks'])} | {result['elapsed_seconds']:.2f}s</div>"
    view.get_root().html.add_child(folium.Element(summary))
    return view


@app.get('/', response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')


@app.get('/app.js')
def ui_script():
    return FileResponse(Path(__file__).parent / 'app.js', media_type='text/javascript')
