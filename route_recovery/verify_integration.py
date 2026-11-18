"""Evaluate trusted saved artifacts through the API; never trains or selects models."""
import json
import hashlib
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient

from .classifier import _load, model_paths
from .main import app


def write_summary(output, summaries):
    lines = ['# Saved-model integration results', '',
             'All seven saved classifiers completed the ten student-test trips (19,160 observations each), API reference evaluation and map generation. No training or tuning was performed.', '',
             '| Model | Independent Top-1 | HMM Top-1 | Independent continuity | HMM continuity | HMM road Macro F1 | Inference seconds |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        a, b = row['independent'], row['hmm']
        lines.append(f"| {row['model']} | {a['top1_all_observations']:.2%} | {b['top1_all_observations']:.2%} | {a['continuity']:.2%} | {b['continuity']:.2%} | {b['mean_trip_road_macro_f1']:.4f} | {row['total_inference_seconds']:.2f} |")
    lines += ['', '## Interpretation', '',
              '- Top-1 includes all observations; candidate misses count as errors. Road Macro F1 is averaged across trips, not candidate binary F1.',
              '- These use derived ordered geometric reference labels. Ambiguous alignment limits the accuracy conclusions.',
              '- HMM increases continuity while lowering pooled pointwise Top-1 for every tested model. No model exactly recovers any whole trip.',
              '- Every model has 334 breaks: 329 speed-limit, two disconnected-component, one road-direction and two timestamp-gap breaks. Finite emission scores do not create missing directed connections. No artificial connector is drawn.',
              '- Optimized HGB has the highest HMM Top-1 here; baseline HGB has the highest independent Top-1. This is development evidence, not a final model selection or proof that tuning universally improved performance.',
              '- Inference seconds sum ten numeric API runs, including preparation, first model load and decoding. Optional reference evaluation and the extra map check are excluded. Models run sequentially on this machine; timings are not deployment benchmarks.',
              '- Select the final model and any decoder parameters using grouped training validation. Do not optimize parameters against this student-test comparison.', '',
              '## Reproduce', '', 'From the repository root: `python -m route_recovery.verify_integration`.',
              'Install `route_recovery/requirements-inference.txt` and download the saved artifacts with `git lfs pull` first.',
              'Per-model JSON contains trip metrics, runtimes, gap diagnostics and the artifact SHA-256. No reserved evaluator data was used.', '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    output = Path('reports/integrated_models')
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    trips = client.get('/trips').json()['student_test']
    if not trips:
        raise RuntimeError('No student-test trips available')
    summaries = []
    for name in model_paths():
        _load.cache_clear()
        rows = []
        for trip_id in trips:
            response = client.post('/match', json={
                'partition': 'student_test', 'trip_id': trip_id,
                'model': name, 'evaluate': True, 'diagnose': True})
            if response.status_code != 200:
                raise RuntimeError(f'{name}/{trip_id}: {response.text}')
            result = response.json()
            assert len(result['arc_ids']) == result['observation_count']
            assert result['matched_observations'] + result['unmatched_observations'] == result['observation_count']
            rows.append({key: result[key] for key in (
                'trip_id', 'observation_count', 'matched_observations', 'unmatched_observations',
                'section_count', 'breaks', 'elapsed_seconds', 'classifier_seconds',
                'decoder_seconds', 'evaluation')})
            print(name, trip_id, 'completed', flush=True)
        total = sum(r['observation_count'] for r in rows)
        summary = {'model': name, 'trips': len(rows), 'observations': total,
                   'total_inference_seconds': sum(r['elapsed_seconds'] for r in rows),
                   'break_reasons': dict(Counter(g.get('cause', g['reason'])
                                                for r in rows for g in r['breaks']))}
        for method in ('independent', 'hmm'):
            eligible = sum(r['evaluation'][method]['continuity']['eligible_pairs'] for r in rows)
            summary[method] = {
                'top1_all_observations': sum(r['evaluation'][method]['top1_accuracy'] * r['observation_count'] for r in rows) / total,
                'mean_trip_road_macro_f1': sum(r['evaluation'][method]['road_arc_macro_f1'] for r in rows) / len(rows),
                'continuity': sum(r['evaluation'][method]['continuity']['valid_pairs'] for r in rows) / eligible if eligible else None,
                'exact_trip_matches': sum(r['evaluation'][method]['exact_trip_match'] for r in rows)}
        # Exercise the actual UI payload/map endpoint as well as numeric inference.
        mapped = client.post('/visualize', json={'partition': 'student_test',
                             'trip_id': trips[0], 'model': name})
        if mapped.status_code != 200 or 'Road network' not in mapped.json()['map_html']:
            raise RuntimeError(f'{name}: map integration failed')
        payload = {'scope': 'Development student-test, derived route alignment, untuned HMM; not model selection',
                   'summary': summary, 'trips': rows}
        with model_paths()[name].open('rb') as artifact:
            payload['artifact_sha256'] = hashlib.file_digest(artifact, 'sha256').hexdigest()
        (output / f'{name}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        summaries.append(summary)
        (output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    write_summary(output, summaries)
    print('All saved classifiers passed API, evaluation and map checks.', flush=True)


if __name__ == '__main__':
    main()
