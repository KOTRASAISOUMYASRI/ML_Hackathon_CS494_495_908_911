import optuna
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupKFold, cross_val_score


def optimize_model(
    model_name,
    X,
    y,
    groups,
    n_trials=10,
    random_state=42,
):
    """Tune a selected classifier using grouped 5-fold CV."""

    if model_name not in {"random_forest", "hist_gradient_boosting"}:
        raise ValueError("Unsupported model name")

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have equal lengths")

    if pd.Series(groups).nunique() < 5:
        raise ValueError("At least 5 unique trips are required")

    cv = GroupKFold(n_splits=5)

    def objective(trial):
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 20, 30, 40]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 5
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.7, 1.0]
                ),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": 1,
            }
            model = RandomForestClassifier(**params)

        else:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.03, 0.2, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
                "max_leaf_nodes": trial.suggest_int(
                    "max_leaf_nodes", 15, 63
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 10, 50
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 1e-4, 10.0, log=True
                ),
                "early_stopping": True,
                "class_weight": "balanced",
                "random_state": random_state,
            }
            model = HistGradientBoostingClassifier(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            error_score="raise",
        )

        trial.set_user_attr("fold_scores", scores.tolist())
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials)

    if model_name == "random_forest":
        best_model = RandomForestClassifier(
            **study.best_params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    else:
        best_model = HistGradientBoostingClassifier(
            **study.best_params,
            early_stopping=True,
            class_weight="balanced",
            random_state=random_state,
        )

    best_model.fit(X, y)

    study.trials_dataframe().to_csv(
        f"{model_name}_optuna_trials.csv", index=False
    )

    return best_model, study
