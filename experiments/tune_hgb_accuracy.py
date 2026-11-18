"""Fast reproducible HGB tuning on complete training trips only.

Select by three-fold grouped road Top-1, retain the baseline as a candidate,
then refit once on all training trips. Student-test is evaluated only afterward.
Existing team artifacts are never overwritten.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from data_loader import discover_trips, load_trip
from feature_engineering import FEATURE_COLUMNS, build_feature_table

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/optimization/hgb_fast'
REPORT=ROOT/'reports/hgb_accuracy_tuning'
BASELINE={'learning_rate':.1,'max_iter':100,'max_leaf_nodes':31,
          'min_samples_leaf':20,'l2_regularization':1.,'class_weight':None}
OLD={'learning_rate':.06562324931873556,'max_iter':180,'max_leaf_nodes':34,
     'min_samples_leaf':38,'l2_regularization':.003776663327107336,'class_weight':'balanced'}


def cache_key(files):
    parts=[(str(p),p.stat().st_size,p.stat().st_mtime_ns)
           for p in (files.track,files.nodes,files.arcs,files.route)]
    for name in ['data_loader.py','feature_engineering.py','preprocessing.py','road_network.py']:
        parts.append(hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:20]


def features(files):
    folder=OUTPUT/'cache';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(cache_key(files)+'.joblib')
    if path.is_file(): return joblib.load(path)
    start=perf_counter()
    result=build_feature_table(load_trip(files,with_reference=True),labeled=True)
    value=(result.candidates,len(result.observations))
    joblib.dump(value,path)
    print('Features',files.trip_id,len(result.candidates),f'{perf_counter()-start:.1f}s',flush=True)
    return value


def measure(table, p, observation_count):
    evaluation=table[['trip_id','observation_id','target']].copy()
    evaluation['p']=p
    best=evaluation.groupby(['trip_id','observation_id'],sort=False)['p'].idxmax()
    correct=int(evaluation.loc[best,'target'].sum())
    return {'top1':correct/observation_count,'correct':correct,'observations':observation_count,
            'candidate_accuracy':accuracy_score(table.target,p>=.5),
            'binary_macro_f1':f1_score(table.target,p>=.5,average='macro',zero_division=0),
            'log_loss':log_loss(table.target,np.column_stack((1-p,p)),labels=[0,1])}


def classifier(params):
    return HistGradientBoostingClassifier(**params,early_stopping=True,random_state=42)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    all_files=discover_trips(ROOT/'train')
    selected=set(pd.Series([f.trip_id for f in all_files]).sample(n=20,random_state=42))
    cached={f.trip_id:features(f) for f in all_files if f.trip_id in selected}
    table=pd.concat([value[0] for value in cached.values()],ignore_index=True)
    counts={trip:value[1] for trip,value in cached.items()}
    folds=list(GroupKFold(n_splits=3).split(table[FEATURE_COLUMNS],table.target,table.trip_id))
    manifest={'partition':'train','search_trips':sorted(selected),'seed':42,'folds':[],
              'objective':'pooled road Top-1 across grouped validation observations, including candidate misses',
              'trials':10,'tpe_startup_trials':3,'student_test_used_for_selection':False}
    for _,valid in folds:
        manifest['folds'].append(sorted(table.iloc[valid].trip_id.unique().tolist()))
    (REPORT/'protocol.json').write_text(json.dumps(manifest,indent=2))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=3))
    study.enqueue_trial(BASELINE);study.enqueue_trial(OLD)
    # Test the old tuned tree settings without forced class balancing.
    study.enqueue_trial({**OLD,'class_weight':None})

    def objective(trial):
        params={'learning_rate':trial.suggest_float('learning_rate',.04,.18,log=True),
                'max_iter':trial.suggest_categorical('max_iter',[100,180,260,320]),
                'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',15,63),
                'min_samples_leaf':trial.suggest_int('min_samples_leaf',10,50),
                'l2_regularization':trial.suggest_float('l2_regularization',.0001,10.,log=True),
                'class_weight':trial.suggest_categorical('class_weight',[None,'balanced'])}
        results=[];start=perf_counter()
        for train,valid in folds:
            fit=table.iloc[train];test=table.iloc[valid]
            model=classifier(params);model.fit(fit[FEATURE_COLUMNS],fit.target)
            p=model.predict_proba(test[FEATURE_COLUMNS])[:,1]
            test=test.reset_index(drop=True)
            n=sum(counts[t] for t in test.trip_id.unique())
            results.append(measure(test,p,n))
        top1=sum(r['correct'] for r in results)/sum(r['observations'] for r in results)
        trial.set_user_attr('fold_metrics',results)
        trial.set_user_attr('cv_macro_f1',float(np.mean([r['binary_macro_f1'] for r in results])))
        trial.set_user_attr('seconds',perf_counter()-start)
        study.trials_dataframe().to_csv(REPORT/'trials.csv',index=False)
        print('TRIAL',trial.number,'Top1',round(top1,6),'MacroF1',trial.user_attrs['cv_macro_f1'],flush=True)
        return top1

    study.optimize(objective,n_trials=manifest['trials'])
    study.trials_dataframe().to_csv(REPORT/'trials.csv',index=False)
    baseline=study.trials[0];previous=study.trials[1];winner=study.best_trial
    summary={'protocol':manifest,'baseline_cv_top1':baseline.value,'previous_tuned_cv_top1':previous.value,
             'winner_cv_top1':winner.value,'winner_cv_macro_f1':winner.user_attrs['cv_macro_f1'],
             'baseline_cv_macro_f1':baseline.user_attrs['cv_macro_f1'],
             'previous_tuned_cv_macro_f1':previous.user_attrs['cv_macro_f1'],
             'winner_trial':winner.number,'winner_params':winner.params,
             'cv_gain_over_baseline_pp':100*(winner.value-baseline.value),
             'cv_gain_over_previous_tuned_pp':100*(winner.value-previous.value)}
    (REPORT/'selection.json').write_text(json.dumps(summary,indent=2))
    print('VALIDATION SELECTION',json.dumps(summary),flush=True)
    tables=[]
    for files in all_files:
        if files.trip_id in cached: tables.append(cached[files.trip_id][0])
        else: tables.append(features(files)[0])
    full=pd.concat(tables,ignore_index=True);del tables
    model=classifier(winner.params);start=perf_counter();model.fit(full[FEATURE_COLUMNS],full.target)
    summary['refit_rows']=len(full);summary['refit_seconds']=perf_counter()-start
    path=OUTPUT/'hist_gradient_boosting_accuracy_tuned.joblib';joblib.dump(model,path)
    del full
    # Fixed winner is evaluated on student-test once; no search decisions follow.
    test_tables=[];observations=0
    for files in discover_trips(ROOT/'student_test'):
        t,n=features(files);test_tables.append(t);observations+=n
    test=pd.concat(test_tables,ignore_index=True);start=perf_counter()
    p=model.predict_proba(test[FEATURE_COLUMNS])[:,1]
    summary['student_test']=measure(test,p,observations)
    summary['student_inference_seconds']=perf_counter()-start
    summary['artifact']=str(path.relative_to(ROOT))
    (REPORT/'selection.json').write_text(json.dumps(summary,indent=2))
    print('FINAL RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
