"""Team-selected classifier and verified saved development comparison."""
import hashlib
import json

from .classifier import ROOT, model_paths

SELECTED_MODEL = 'hist_gradient_boosting_optimized'
SELECTED_LABEL = 'HistGradientBoosting (previous F1 tuned)'


def selected_comparison():
    result = {'model': SELECTED_MODEL, 'label': SELECTED_LABEL,
              'selection_source': 'Team choice of HistGradientBoosting; optimized artifact',
              'report_available': False,
              'scope': 'Ten development student-test trips; derived reference alignment; not reserved evaluation'}
    report = ROOT / 'reports' / 'integrated_models' / f'{SELECTED_MODEL}.json'
    artifact = model_paths().get(SELECTED_MODEL)
    if not report.is_file() or artifact is None:
        result['message'] = 'Saved comparison or selected artifact is unavailable. Run integration verification.'
        return result
    try:
        data = json.loads(report.read_text(encoding='utf-8'))
        with artifact.open('rb') as stream:
            fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
        if fingerprint != data['artifact_sha256'] or data['summary']['model'] != SELECTED_MODEL:
            raise ValueError('Artifact differs from the measured comparison')
        summary = data['summary']
        independent, hmm = summary['independent'], summary['hmm']
        changes = {'top1_percentage_points': 100 * (hmm['top1_all_observations'] - independent['top1_all_observations']),
                   'continuity_percentage_points': 100 * (hmm['continuity'] - independent['continuity'])}
    except (OSError, ValueError, KeyError, TypeError):
        result['message'] = 'Saved comparison is invalid or stale. Re-run integration verification.'
        return result
    result.update(report_available=True, summary=summary, changes=changes,
                  artifact_sha256=fingerprint,
                  message='HMM improves continuity but lowers pooled Top-1. Report both outcomes.')
    return result
