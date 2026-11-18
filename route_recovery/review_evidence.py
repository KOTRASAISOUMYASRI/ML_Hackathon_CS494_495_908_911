"""Measure saved HGB artifacts on identical development trips for review evidence.

No training, tuning, reserved-data use or application configuration changes.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay)

from data_loader import discover_trips, load_trip
from feature_engineering import _align_reference
from .classifier import ROOT, _load, model_paths, score_candidates
from .data_adapter import SharedRoadTrip


def main():
    output = ROOT / 'reports' / 'review'
    output.mkdir(parents=True, exist_ok=True)
    names = ['hist_gradient_boosting', 'hist_gradient_boosting_optimized']
    paths = model_paths()
    probabilities = {name: [] for name in names}
    targets, rows, trip_rows = [], [], []
    # Features and probabilities are computed without reference access.
    for files in discover_trips(ROOT / 'student_test'):
        trip = SharedRoadTrip(files)
        scores = {name: score_candidates(trip, name)[1] for name in names}
        reference = load_trip(files, with_reference=True)
        points = np.array([trip.project(*o[:2]) for o in trip.track])
        labels = _align_reference(points, trip.network, reference.reference_arcs)
        y = np.array([int(c.arc_id == label) for label, candidates in
                      zip(labels, trip.candidate_rows) for c in candidates])
        targets.append(y)
        for name in names:
            probabilities[name].append(scores[name])
            offset = correct = 0
            for label, candidates in zip(labels, trip.candidate_rows):
                count = len(candidates)
                if count:
                    best = int(np.argmax(scores[name][offset:offset+count]))
                    correct += candidates[best].arc_id == label
                offset += count
            trip_rows.append({'model': name, 'trip_id': files.trip_id,
                              'observations': len(labels), 'correct_top1': int(correct)})
        print(files.trip_id, 'both artifacts scored', flush=True)
    y = np.concatenate(targets)
    parameters = {}
    for name, path in paths.items():
        model = _load(str(path.resolve()), path.stat().st_mtime_ns)
        classifier = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
        record = {'classifier': type(classifier).__name__, 'parameters': classifier.get_params(deep=False)}
        if hasattr(model, 'named_steps'):
            record['scaler'] = type(model.named_steps['scaler']).__name__
        if hasattr(classifier, 'n_iter_'):
            record['actual_iterations'] = classifier.n_iter_
        parameters[name] = record
    (output / 'saved_hyperparameters.json').write_text(json.dumps(parameters, indent=2, default=str), encoding='utf-8')
    for name in names:
        p = np.concatenate(probabilities[name]); prediction = p >= .5
        groups = [r for r in trip_rows if r['model'] == name]
        with paths[name].open('rb') as artifact:
            fingerprint = hashlib.file_digest(artifact, 'sha256').hexdigest()
        row = {'model': name, 'candidate_rows': len(y),
               'observations': sum(r['observations'] for r in groups),
               'candidate_accuracy': accuracy_score(y, prediction),
               'positive_precision': precision_score(y, prediction, zero_division=0),
               'positive_recall': recall_score(y, prediction, zero_division=0),
               'positive_f1': f1_score(y, prediction, zero_division=0),
               'binary_macro_f1': f1_score(y, prediction, labels=[0,1], average='macro', zero_division=0),
               'log_loss': log_loss(y, np.column_stack((1-p,p)), labels=[0,1]),
               'top1_all_observations': sum(r['correct_top1'] for r in groups) / sum(r['observations'] for r in groups),
               'artifact_sha256': fingerprint,
               'confusion_matrix': confusion_matrix(y, prediction, labels=[0,1]).tolist()}
        rows.append(row)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ConfusionMatrixDisplay(np.array(row['confusion_matrix']), display_labels=['Incorrect candidate', 'Correct candidate']).plot(ax=ax, values_format='d', colorbar=False)
        ax.set_title(('Tuned' if name.endswith('_optimized') else 'Baseline') + ' HGB: student-test candidate classification')
        fig.tight_layout(); fig.savefig(output / f'{name}_confusion_matrix.png', dpi=160); plt.close(fig)
    payload = {'scope': 'Ten student-test development trips; identical derived labels and candidates; no retraining',
               'positive_candidates': int(y.sum()), 'negative_candidates': int(len(y)-y.sum()),
               'models': rows, 'trips': trip_rows}
    (output / 'hgb_tuning_comparison.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if k != 'confusion_matrix'} for r in rows]).to_csv(output / 'hgb_tuning_comparison.csv', index=False)
    metrics = ['candidate_accuracy','positive_f1','top1_all_observations']
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(metrics)); width=.35
    for i,r in enumerate(rows):
        bars=ax.bar(x+(i-.5)*width, [r[k] for k in metrics], width, label=['Before tuning','After tuning'][i])
        ax.bar_label(bars, labels=[f'{r[k]:.2%}' for k in metrics], padding=3)
    ax.set_xticks(x, ['Candidate accuracy','Positive-class F1','Road Top-1 (all observations)'])
    ax.set_ylim(0,1.12); ax.set_ylabel('Score'); ax.legend(loc='lower right')
    ax.set_title('Saved HGB before/after tuning: same ten student-test trips')
    fig.tight_layout(); fig.savefig(output / 'hgb_before_after_tuning.png', dpi=160); plt.close(fig)
    measured=json.loads((ROOT/'reports/integrated_models/hist_gradient_boosting_optimized.json').read_text())['summary']
    assert abs(rows[1]['top1_all_observations']-measured['independent']['top1_all_observations']) < 1e-12
    fig, ax=plt.subplots(figsize=(8,5)); x=np.arange(2)
    for i,method in enumerate(['independent','hmm']):
        m=measured[method]; values=[m['top1_all_observations'],m['continuity']]
        bars=ax.bar(x+(i-.5)*width,values,width,label=['Classifier alone','Classifier + HMM'][i])
        ax.bar_label(bars,labels=[f'{v:.2%}' for v in values],padding=3)
    ax.set_xticks(x,['Road Top-1 accuracy','Eligible-transition continuity']); ax.set_ylim(0,1.12)
    ax.set_ylabel('Fraction'); ax.legend(loc='lower right'); ax.set_title('Optimized HGB before/after HMM: ten student-test trips')
    fig.tight_layout(); fig.savefig(output/'hgb_before_after_hmm.png',dpi=160); plt.close(fig)
    trials=pd.read_csv(ROOT/'outputs/optimization/hist_gradient_boosting_trials.csv')
    fig, ax=plt.subplots(figsize=(8,4.5)); ax.plot(trials['number'],trials['value'],'o-',label='Trial CV F1')
    ax.plot(trials['number'],trials['value'].cummax(),'s--',label='Best so far')
    ax.set_xticks(trials['number']); ax.set_xlabel('Optuna trial number'); ax.set_ylabel('Mean positive-class F1 (3 grouped folds)')
    ax.set_title('HGB tuning: four trials on 20 complete training trips'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'hgb_optimization_trials.png',dpi=160); plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)


if __name__ == '__main__':
    main()
