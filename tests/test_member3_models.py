import numpy as np
from pathsnap.models import (
    create_random_forest,
    create_hist_gradient_boosting,
)

def test_both_models_follow_sklearn_interface():
    rng = np.random.default_rng(42)
    X = rng.random((100, 11))
    y = np.array([0, 1] * 50)

    models = [
        create_random_forest(),
        create_hist_gradient_boosting(),
    ]

    for model in models:
        model.fit(X, y)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        assert predictions.shape == (100,)
        assert probabilities.shape == (100, 2)
        assert np.all(np.isfinite(probabilities))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
