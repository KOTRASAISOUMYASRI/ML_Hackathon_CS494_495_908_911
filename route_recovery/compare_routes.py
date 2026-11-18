"""Reproducible classifier/HMM development comparison, without model training."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from data_loader import discover_trips
from .classifier import score_candidates
from .data_adapter import SharedRoadTrip
from .evaluation import evaluate_reference
from .hmm import recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('partition', choices=['train', 'student_test'])
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--model', default='decision_tree')
    parser.add_argument('--output', default='reports/member4_classifier_hmm.json')
    args = parser.parse_args()
    rows = []
    for files in discover_trips(Path(args.data_root) / args.partition):
        start = perf_counter()
        trip = SharedRoadTrip(files)
        scorer, p = score_candidates(trip, args.model)
        result = recover(trip, scorer=scorer)
        inference_seconds = perf_counter() - start
        metrics = evaluate_reference(trip, result, files, p)
        rows.append({'trip_id': files.trip_id, 'inference_seconds': inference_seconds,
                     'breaks': len(result['breaks']), 'evaluation': metrics})
        print(files.trip_id, metrics['independent']['top1_accuracy'], metrics['hmm']['top1_accuracy'], flush=True)
    total = sum(row['evaluation']['observations'] for row in rows)
    summary = {}
    for method in ('independent', 'hmm'):
        eligible = sum(r['evaluation'][method]['continuity']['eligible_pairs'] for r in rows)
        summary[method] = {
            'top1_accuracy_all_observations': sum(r['evaluation'][method]['top1_accuracy'] * r['evaluation']['observations'] for r in rows) / total,
            'mean_trip_road_arc_macro_f1': sum(r['evaluation'][method]['road_arc_macro_f1'] for r in rows) / len(rows),
            'continuity': sum(r['evaluation'][method]['continuity']['valid_pairs'] for r in rows) / eligible if eligible else None,
            'exact_trip_matches': sum(r['evaluation'][method]['exact_trip_match'] for r in rows)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'model': args.model, 'partition': args.partition,
        'observations': total, 'scope': 'Existing baseline artifact, untuned decoder, derived reference labels; not final model selection',
        'decoder': {'beta_m':30, 'max_gap_s':30, 'max_speed_mps':60},
        'summary': summary, 'trips': rows}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
