
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from models.random_forest import create_random_forest
from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Processed Dataset" / "outputs"
OUTPUT = ROOT / "outputs" / "models"

def main():
    print("Loading datasets...")

    train = pd.read_csv(
        DATA / "train" / "candidates.csv",
        dtype={"trip_id": str},
    )
    test = pd.read_csv(
        DATA / "student_test" / "candidates.csv",
        dtype={"trip_id": str},
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    model = create_random_forest()

    print("\nTraining Random Forest...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    print("Training completed.")

    print("\nEvaluating...")
    start = time.perf_counter()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "log_loss": log_loss(y_test, probabilities),
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
    }

    # Top-1 accuracy: choose the highest-probability candidate per observation.
    results = test[
        ["trip_id", "observation_id", TARGET_COLUMN]
    ].copy()
    results["probability"] = probabilities

    top_indices = results.groupby(
        ["trip_id", "observation_id"]
    )["probability"].idxmax()

    top1_accuracy = results.loc[top_indices, TARGET_COLUMN].mean()
    metrics["top1_accuracy"] = top1_accuracy

    # Candidate misses are observations with no candidate rows.
    observations = pd.read_csv(
        DATA / "student_test" / "observations.csv",
        dtype={"trip_id": str},
    )
    metrics["candidate_misses"] = int(
        observations["candidate_miss"].sum()
    )
    metrics["total_observations"] = len(observations)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT / "random_forest.joblib")

    pd.DataFrame([metrics]).to_csv(
        OUTPUT / "random_forest_metrics.csv",
        index=False,
    )

    print("\nRANDOM FOREST RESULTS")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nModel saved to:", OUTPUT / "random_forest.joblib")


if __name__ == "__main__":
    main()
