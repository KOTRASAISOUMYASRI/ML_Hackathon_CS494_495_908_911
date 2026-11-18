"""Read-only adapters for trusted, server-configured sklearn artifacts."""
import os
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from feature_engineering import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def model_paths():
    paths = {name: ROOT / 'models' / f'{name}.joblib'
             for name in ('decision_tree', 'logistic_regression', 'knn')}
    for name in ('random_forest', 'hist_gradient_boosting'):
        paths[name] = ROOT / 'outputs' / 'models' / f'{name}.joblib'
        paths[name + '_optimized'] = ROOT / 'outputs' / 'optimization' / f'{name}_optimized.joblib'
    paths['hist_gradient_boosting_candidate_tuned'] = (
        ROOT / 'outputs/optimization/hgb_fast/hist_gradient_boosting_candidate_tuned.joblib')
    if os.environ.get('PATHSNAP_MODEL_ARTIFACT'):
        paths['configured'] = Path(os.environ['PATHSNAP_MODEL_ARTIFACT'])
    return {name: path for name, path in paths.items() if path.is_file()}


@lru_cache(maxsize=1)
def _load(path, modified_ns):
    # Never accept an artifact path from an HTTP request. Only trusted team files.
    with open(path, 'rb') as artifact:
        if artifact.read(43).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Model is a Git LFS pointer. Run git lfs pull on the server.')
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        try:
            model = joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise ValueError('Model/library version mismatch. Use the sklearn version used for training.') from exc
    if not hasattr(model, 'predict_proba') or set(model.classes_) != {0, 1}:
        raise ValueError('Artifact must expose predict_proba with binary classes 0 and 1')
    if list(getattr(model, 'feature_names_in_', [])) != FEATURE_COLUMNS:
        raise ValueError('Artifact feature names/order must equal shared FEATURE_COLUMNS')
    return model


def score_candidates(trip, name):
    path = model_paths().get(name)
    if path is None:
        raise ValueError('Unknown or unavailable classifier')
    model = _load(str(path.resolve()), path.stat().st_mtime_ns)
    table = trip.features.candidates
    raw = (np.asarray(model.predict_proba(table[FEATURE_COLUMNS]), dtype=float)
           if len(table) else np.empty((0, 2)))
    if raw.shape != (len(table), 2) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError('Classifier returned invalid candidate probabilities')
    probabilities = raw[:, list(model.classes_).index(1)]
    groups, offset = [], 0
    for candidates in trip.candidate_rows:
        count = len(candidates)
        if count:
            values = np.clip(probabilities[offset:offset + count], 1e-12, 1)
            groups.append((tuple(c.arc_id for c in candidates), np.log(values / values.sum()).tolist()))
        offset += count
    if offset != len(table):
        raise ValueError('Candidate table and decoder rows differ')
    iterator = iter(groups)

    def scorer(observation, candidates):
        ids, scores = next(iterator)
        if ids != tuple(c.arc_id for c in candidates):
            raise ValueError('Candidate order differs from classifier feature rows')
        return scores

    return scorer, probabilities
