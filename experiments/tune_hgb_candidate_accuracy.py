"""Candidate-accuracy search; all selection uses training-trip validation.

This is a distinct objective from road Top-1. Student-test is reused development
evaluation, not an untouched final test. Existing models are preserved.
"""
import json
import hashlib
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from data_loader import discover_trips
from feature_engineering import FEATURE_COLUMNS
from experiments.tune_hgb_accuracy import ROOT, OUTPUT, BASELINE, features, classifier, measure


def main():
    report=ROOT/'reports/hgb_candidate_accuracy_tuning';report.mkdir(parents=True,exist_ok=True)
    files=discover_trips(ROOT/'train')
    ids=set(pd.Series([f.trip_id for f in files]).sample(n=20,random_state=42))
    cached={f.trip_id:features(f) for f in files if f.trip_id in ids}
    table=pd.concat([v[0] for v in cached.values()],ignore_index=True)
    counts={k:v[1] for k,v in cached.items()}
    folds=list(GroupKFold(3).split(table[FEATURE_COLUMNS],table.target,table.trip_id))
    protocol={'objective':'pooled candidate binary accuracy at threshold 0.5',
              'selection_partition':'train','search_trips':sorted(ids),'folds':3,'seed':42,
              'trials':8,'startup_trials':3,'class_weight':None,
              'reason_for_unweighted_search':'unweighted baseline has higher candidate accuracy and Macro F1 in first training CV experiment',
              'student_test_used_for_trial_selection':False,'student_test_scope':'reused development evaluation; not untouched final test'}
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=3))
    base={k:v for k,v in BASELINE.items() if k!='class_weight'}
    study.enqueue_trial(base)
    study.enqueue_trial({**base,'max_iter':180,'max_leaf_nodes':47,'l2_regularization':.1})
    study.enqueue_trial({**base,'learning_rate':.08,'max_iter':260,'max_leaf_nodes':63,'min_samples_leaf':15,'l2_regularization':.1})

    def objective(trial):
        params={'learning_rate':trial.suggest_float('learning_rate',.06,.16,log=True),
                'max_iter':trial.suggest_categorical('max_iter',[100,180,260]),
                'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',25,63),
                'min_samples_leaf':trial.suggest_int('min_samples_leaf',10,45),
                'l2_regularization':trial.suggest_float('l2_regularization',.001,5.,log=True)}
        rows=[];candidate_counts=[];start=perf_counter()
        for train,valid in folds:
            fit=table.iloc[train];test=table.iloc[valid].reset_index(drop=True)
            model=classifier({**params,'class_weight':None});model.fit(fit[FEATURE_COLUMNS],fit.target)
            p=model.predict_proba(test[FEATURE_COLUMNS])[:,1]
            rows.append(measure(test,p,sum(counts[t] for t in test.trip_id.unique())))
            candidate_counts.append(len(test))
        value=float(np.average([r['candidate_accuracy'] for r in rows],weights=candidate_counts))
        trial.set_user_attr('fold_metrics',rows)
        trial.set_user_attr('cv_macro_f1',float(np.mean([r['binary_macro_f1'] for r in rows])))
        trial.set_user_attr('cv_top1',sum(r['correct'] for r in rows)/sum(r['observations'] for r in rows))
        trial.set_user_attr('seconds',perf_counter()-start)
        print('CANDIDATE TRIAL',trial.number,'Accuracy',value,'Top1',trial.user_attrs['cv_top1'],flush=True)
        return value

    study.optimize(objective,n_trials=8,callbacks=[lambda s,t:s.trials_dataframe().to_csv(report/'trials.csv',index=False)])
    baseline=study.trials[0];best=study.best_trial
    summary={'protocol':protocol,'baseline_cv_accuracy':baseline.value,'best_cv_accuracy':best.value,
             'baseline_cv_top1':baseline.user_attrs['cv_top1'],'best_cv_top1':best.user_attrs['cv_top1'],
             'baseline_cv_macro_f1':baseline.user_attrs['cv_macro_f1'],'best_cv_macro_f1':best.user_attrs['cv_macro_f1'],
             'params':{**best.params,'class_weight':None},'best_trial':best.number}
    (report/'results.json').write_text(json.dumps(summary,indent=2))
    full=pd.concat([cached[f.trip_id][0] if f.trip_id in cached else features(f)[0] for f in files],ignore_index=True)
    model=classifier(summary['params']);start=perf_counter();model.fit(full[FEATURE_COLUMNS],full.target)
    summary['refit_rows']=len(full);summary['refit_seconds']=perf_counter()-start
    artifact=OUTPUT/'hist_gradient_boosting_candidate_tuned.joblib';joblib.dump(model,artifact)
    del full
    test_tables=[];observations=0
    for f in discover_trips(ROOT/'student_test'):
        t,n=features(f);test_tables.append(t);observations+=n
    test=pd.concat(test_tables,ignore_index=True);start=perf_counter()
    summary['student_test']=measure(test,model.predict_proba(test[FEATURE_COLUMNS])[:,1],observations)
    summary['student_inference_seconds']=perf_counter()-start
    summary['artifact']=artifact.relative_to(ROOT).as_posix()
    with artifact.open('rb') as stream:
        summary['artifact_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    summary['sklearn_version']=sklearn.__version__
    summary['feature_columns']=FEATURE_COLUMNS
    (report/'results.json').write_text(json.dumps(summary,indent=2))
    print('FINAL CANDIDATE ACCURACY RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4): main()
