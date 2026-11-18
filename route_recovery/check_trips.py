"""Reproducible execution checks, not reference-based accuracy evaluation."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from data_loader import discover_trips
from .data_adapter import SharedRoadTrip
from .hmm import recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('partition')
    parser.add_argument('--output', default='reports/student_test_execution.json')
    parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args()
    rows = []
    for files in discover_trips(args.partition):
        started = perf_counter()
        row = {'trip_id': files.trip_id}
        try:
            trip = SharedRoadTrip(files)
            result = recover(trip, diagnose=args.diagnose)
            count = len(trip.track)
            assert len(result['arc_ids']) == count
            assert len(result['observation_ids']) == count
            assert all(arc is None or 0 <= arc < len(trip.arcs) for arc in result['arc_ids'])
            assert sum(len(section) > 0 for section in result['route_sections']) == len(result['route_sections'])
            row.update(status='passed', observations=count,
                       matched=result['matched_observations'],
                       sections=len(result['route_sections']),
                       breaks=dict(Counter(b['reason'] for b in result['breaks'])))
            if args.diagnose:
                row['causes'] = dict(Counter(b.get('cause', b['reason']) for b in result['breaks']))
                row['break_details'] = result['breaks']
                row['fallback_observations'] = sum(min(c.distance_m for c in cs) > 50 for cs in trip.candidate_rows)
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['seconds'] = round(perf_counter() - started, 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Execution and output-shape checks only; no accuracy claims or reference reads.',
            'configuration': {'radius_m': 50, 'max_candidates': 8, 'sigma_m': 20,
                              'beta_m': 30, 'max_gap_s': 30, 'max_speed_mps': 60},
            'trips': rows}, indent=2), encoding='utf-8')
    if any(row['status'] == 'failed' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
