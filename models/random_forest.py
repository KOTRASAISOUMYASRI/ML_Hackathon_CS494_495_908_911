from sklearn.ensemble import RandomForestClassifier

def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create the Random Forest baseline classifier."""
    return RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
