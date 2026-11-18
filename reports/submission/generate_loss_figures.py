"""Plot stored HGB loss histories and explicit CV evidence coverage."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from route_recovery.classifier import model_paths, _load


def main():
    output=Path('reports/submission'); histories={}
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['hist_gradient_boosting','hist_gradient_boosting_optimized']):
        path=model_paths()[name]; model=_load(str(path.resolve()),path.stat().st_mtime_ns)
        train=-np.asarray(model.train_score_); validation=-np.asarray(model.validation_score_)
        ax.plot(train,label='Internal training loss'); ax.plot(validation,label='Internal validation loss')
        ax.set_title('Tuned HGB' if name.endswith('optimized') else 'Baseline HGB')
        ax.set_xlabel('Boosting iteration');ax.set_ylabel('Log loss');ax.legend(fontsize=8)
        histories[name]={'training_loss':train.tolist(),'validation_loss':validation.tolist()}
    fig.tight_layout();fig.savefig(output/'hgb_actual_loss_histories.png',dpi=160);plt.close(fig)
    (output/'hgb_actual_loss_histories.json').write_text(json.dumps(histories,indent=2))
    fig,ax=plt.subplots(figsize=(9,3.5))
    # Missing scores are annotated, never represented as measured zeros.
    ax.bar([2],[0.8344550163987072],color='#167a84',width=.5)
    for index in [0,1,3,4]:
        ax.text(index,.08,'Not recorded',ha='center',rotation=90,color='#8b3a26')
    ax.set_xticks(np.arange(5),['LR','DT','KNN','RF','HGB']);ax.set_xlim(-.6,4.6);ax.set_ylim(0,1)
    ax.set_ylabel('CV binary Macro F1');ax.set_title('CV Macro F1 evidence coverage incomplete for five-model comparison')
    ax.text(2,.86,'0.8345',ha='center');fig.tight_layout();fig.savefig(output/'cv_macro_f1_evidence.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
