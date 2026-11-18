"""Summarize both tuning attempts without overwriting historical reports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'reports/hgb_candidate_accuracy_tuning'
    new=json.loads((folder/'results.json').read_text())
    old=json.loads((root/'reports/review/hgb_tuning_comparison.json').read_text())['models']
    previous=old[1];baseline=old[0];current=new['student_test']
    rows=[('Untuned HGB',baseline['candidate_accuracy'],baseline['top1_all_observations'],baseline['binary_macro_f1'],baseline['log_loss']),
          ('Previous F1 tuned',previous['candidate_accuracy'],previous['top1_all_observations'],previous['binary_macro_f1'],previous['log_loss']),
          ('New accuracy tuned',current['candidate_accuracy'],current['top1'],current['binary_macro_f1'],current['log_loss'])]
    lines=['# HGB accuracy tuning results','',
           'Two training-validation searches were performed. Existing team models were preserved. The new candidate-accuracy model improves over the previous F1-tuned artifact, but does not beat the original untuned baseline on the development partition.','',
           '| Model | Candidate accuracy | Road Top-1 | Candidate Macro F1 | Log loss |',
           '|---|---:|---:|---:|---:|']
    for label,accuracy,top1,macro,loss in rows:
        lines.append(f'| {label} | {accuracy:.2%} | {top1:.2%} | {macro:.4f} | {loss:.4f} |')
    lines += ['',f"Gain over the previous tuned artifact: {100*(current['candidate_accuracy']-previous['candidate_accuracy']):+.2f} percentage points candidate accuracy and {100*(current['top1']-previous['top1_all_observations']):+.2f} points road Top-1.",
              f"Difference from the untuned baseline: {100*(current['candidate_accuracy']-baseline['candidate_accuracy']):+.2f} points candidate accuracy and {100*(current['top1']-baseline['top1_all_observations']):+.2f} points road Top-1.",'',
              '## Protocol','',
              'Search 1: ten trials, three TPE startup trials, followed by seven adaptive trials; objective grouped road Top-1. It produced a small CV gain but worse development results, and was not promoted.',
              'Search 2: eight trials, three startup trials, followed by five adaptive trials; objective candidate accuracy at probability threshold 0.5. Unweighted recipes were studied because the first training-CV experiment favored them for candidate accuracy. The winning trial was selected before its development evaluation.',
              'Both use the same 20 complete training trips and three GroupKFold splits, then refit on all 80 training trips (1,103,062 candidate rows). No student-test rows enter fitting or trial scoring. Student-test has been reused for development comparisons and is not an untouched final test. The reserved evaluator partition is unused.','',
              f"Grouped CV candidate accuracy: {new['baseline_cv_accuracy']:.4%} baseline -> {new['best_cv_accuracy']:.4%} selected.",
              f"Grouped CV road Top-1: {new['baseline_cv_top1']:.4%} -> {new['best_cv_top1']:.4%}.",
              f"Grouped CV Macro F1: {new['baseline_cv_macro_f1']:.6f} -> {new['best_cv_macro_f1']:.6f}.",
              'The validation gains are very small; no statistical significance is established. Candidate accuracy improved while CV Macro F1 slightly decreased.','',
              '## Saved settings','', '```json',json.dumps(new['params'],indent=2),'```','',
              'Artifact: `'+new['artifact']+'`. Existing artifacts are unchanged.',
              'The new model is exposed as `hist_gradient_boosting_candidate_tuned` / HistGradientBoosting (accuracy tuned) after restarting the API. The selected default remains the previous artifact, so the historical HMM comparison remains correctly attributed.','',
              'A real trip-52 /visualize test verified loading, features, HMM, optional evaluation and map generation; see api_check.json. The all-ten-trip accuracy values above are classifier-only. No pooled HMM improvement is claimed for the new model.','',
              '## Reproduce','',
              '`python -m experiments.tune_hgb_accuracy`',
              '`python -m experiments.tune_hgb_candidate_accuracy`',
              '`python -m experiments.report_hgb_accuracy_tuning`','',
              'Trial logs, results and the API check are in this directory and ../hgb_accuracy_tuning/. Cache files stay under ignored outputs/. All observations, including candidate misses, count in Top-1.','']
    (folder/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,index,title in zip(axes,[1,2],['Candidate classification accuracy','Road Top-1 all observations']):
        values=[r[index] for r in rows]
        bars=ax.bar(['Untuned','Previous tuned','New tuned'],values,color=['#546e7a','#dd8452','#167a84'])
        ax.bar_label(bars,labels=[f'{v:.2%}' for v in values],padding=4)
        ax.set_ylim(0,1.08);ax.set_title(title);ax.set_ylabel('Fraction')
    fig.suptitle('Same ten development trips; untuned baseline still has the highest accuracy')
    fig.tight_layout();fig.savefig(folder/'accuracy_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
