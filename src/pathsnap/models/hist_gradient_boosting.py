from sklearn.ensemble import HistGradientBoostingClassifier

def create_hist_gradient_boosting(
    random_state: int = 42
) -> HistGradientBoostingClassifier:
    """Create the HistGradientBoosting baseline classifier."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=random_state
    )
