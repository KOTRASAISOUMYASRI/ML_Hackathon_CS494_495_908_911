"""Recompute common held-out metrics from saved classifiers without training."""
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, log_loss, accuracy_score

from data_loader import discover_trips, load_trip
from feature_engineering import _align_reference
from route_recovery.classifier import score_candidates, model_paths, _load
from route_recovery.data_adapter import SharedRoadTrip


def main():
    output = Path('reports/submission')
    trips=[]; targets=[]
    for files in discover_trips(Path('student_test')):
        trip=SharedRoadTrip(files)
        reference=load_trip(files,with_reference=True)
        points=np.array([trip.project(*o[:2]) for o in trip.track])
        labels=_align_reference(points,trip.network,reference.reference_arcs)
        targets.append(np.array([int(c.arc_id==label) for label,row in zip(labels,trip.candidate_rows) for c in row]))
        trips.append(trip)
    y=np.concatenate(targets); rows=[]
    for name in model_paths():
        if name=='configured':
            continue
        _load.cache_clear(); probabilities=[]; seconds=0
        for trip in trips:
            start=perf_counter(); p=score_candidates(trip,name)[1]
            seconds+=perf_counter()-start; probabilities.append(p)
        p=np.concatenate(probabilities)
        rows.append({'model':name,'candidate_rows':len(y),'candidate_accuracy':accuracy_score(y,p>=.5),
                     'student_binary_macro_f1':f1_score(y,p>=.5,average='macro',zero_division=0),
                     'log_loss':log_loss(y,np.column_stack((1-p,p)),labels=[0,1]),
                     'classifier_seconds_including_first_load':seconds})
        print(name,rows[-1],flush=True)
    (output/'common_student_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    route={r['model']:r for r in json.loads(Path('reports/integrated_models/summary.json').read_text())}
    labels=[r['model'].replace('hist_gradient_boosting','HGB').replace('logistic_regression','LR').replace('random_forest','RF').replace('decision_tree','DT').replace('_optimized',' tuned') for r in rows]
    x=np.arange(len(rows)); width=.36
    fig,ax=plt.subplots(figsize=(10,4.8))
    for i,(metric,title) in enumerate([('top1','Road Top-1'),('macro','Candidate Macro F1')]):
        values=[route[r['model']]['independent']['top1_all_observations'] if metric=='top1' else r['student_binary_macro_f1'] for r in rows]
        bars=ax.bar(x+(i-.5)*width,values,width,label=title)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=2)
    ax.set_xticks(x,labels,rotation=18); ax.set_ylim(0,1.12); ax.set_ylabel('Score'); ax.legend(loc='lower right'); ax.set_title('Common student-test evaluation across saved classifiers')
    fig.tight_layout(); fig.savefig(output/'student_comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.4)); bars=ax.bar(labels,[r['log_loss'] for r in rows],color='#167a84')
    ax.bar_label(bars,fmt='%.3f',padding=3); ax.tick_params(axis='x',labelrotation=18)
    ax.set_ylabel('Candidate log loss'); ax.set_title('Student-test log loss lower is better')
    fig.tight_layout(); fig.savefig(output/'log_loss_comparison.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
