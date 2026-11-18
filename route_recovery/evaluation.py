"""Aggregate reference evaluation AFTER inference; labels never leave the server."""
import numpy as np
from sklearn.metrics import f1_score, log_loss

from data_loader import load_trip
from feature_engineering import _align_reference


def continuity(trip, predictions, gap_s=30, speed_mps=60):
    eligible = valid = 0
    previous = None
    for t, arc in enumerate(predictions):
        current = next((c for c in trip.candidate_rows[t] if c.arc_id == arc), None)
        if t:
            dt = trip.track[t][2] - trip.track[t-1][2]
            if trip.observation_ids[t] == trip.observation_ids[t-1] + 1 and dt <= gap_s:
                eligible += 1
                if previous is not None and current is not None:
                    distance, _ = trip.connect(previous, current, dt * speed_mps)
                    valid += np.isfinite(distance) and distance <= dt * speed_mps
        previous = current
    return {'valid_pairs': int(valid), 'eligible_pairs': eligible,
            'fraction': valid / eligible if eligible else None}


def evaluate_reference(trip, result, files, probabilities=None):
    reference = load_trip(files, with_reference=True)
    points = np.array([trip.project(*o[:2]) for o in trip.track])
    labels = _align_reference(points, trip.network, reference.reference_arcs)
    misses = sum(int(label) not in {c.arc_id for c in row}
                 for label, row in zip(labels, trip.candidate_rows))
    metrics = {'reference_method': 'Derived ordered geometric alignment; labels may be ambiguous',
               'observations': len(labels), 'candidate_misses': misses,
               'candidate_miss_rate': misses / len(labels)}
    for name, key in [('independent', 'independent_arc_ids'), ('hmm', 'arc_ids')]:
        prediction = np.array([-1 if arc is None else arc for arc in result[key]])
        correct = prediction == labels
        metrics[name] = {
            'top1_accuracy': float(correct.mean()),
            'road_arc_macro_f1': float(f1_score(labels, prediction, average='macro', zero_division=0)),
            'exact_trip_match': bool(correct.all()), 'continuity': continuity(trip, result[key])}
    if probabilities is not None:
        targets = np.array([int(c.arc_id == label) for label, row in zip(labels, trip.candidate_rows) for c in row])
        metrics['candidate'] = {
            'macro_f1': float(f1_score(targets, probabilities >= .5, labels=[0, 1], average='macro', zero_division=0)),
            'log_loss': float(log_loss(targets, np.column_stack((1-probabilities, probabilities)), labels=[0, 1]))}
    return metrics
