
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
    n_trials=4,
    n_splits=3,
    refit_X=None,
    refit_y=None,
    random_state=42,
):
    """Tune on a trip-level subset and optionally refit on all training data."""

    if model_name not in {"random_forest", "hist_gradient_boosting"}:
        raise ValueError("Unsupported model name")

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have equal lengths")

    if pd.Series(groups).nunique() < n_splits:
        raise ValueError("Not enough unique trips for cross-validation")

    cv = GroupKFold(n_splits=n_splits)

    def objective(trial):
        if model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=50),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [15, 25, None]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 8
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 2, 8
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.7]
                ),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": 1,
            }
            model = RandomForestClassifier(**params)

        else:
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.04, 0.15, log=True
                ),
                "max_iter": trial.suggest_int("max_iter", 80, 180, step=50),
                "max_leaf_nodes": trial.suggest_int(
                    "max_leaf_nodes", 15, 40
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 20, 50
                ),
                "l2_regularization": trial.suggest_float(
                    "l2_regularization", 0.001, 5.0, log=True
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

    final_X = refit_X if refit_X is not None else X
    final_y = refit_y if refit_y is not None else y

    final_params = study.best_params.copy()
    final_params.update({
        "class_weight": "balanced",
        "random_state": random_state,
    })

    if model_name == "random_forest":
        final_params["n_jobs"] = 1
        best_model = RandomForestClassifier(**final_params)
    else:
        final_params["early_stopping"] = True
        best_model = HistGradientBoostingClassifier(**final_params)

    best_model.fit(final_X, final_y)

    study.trials_dataframe().to_csv(
        f"{model_name}_optuna_trials.csv", index=False
    )

    return best_model, study
