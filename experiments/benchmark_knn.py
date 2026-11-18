
from pathlib import Path
import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

from models.knn import build_model


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "outputs" / "train" / "candidates.csv"
RESULTS_DIR = ROOT / "results"

FEATURE_COLUMNS = [
    "distance_to_road_m",
    "along_segment_fraction",
    "segment_length_m",
    "heading_error_deg",
    "gps_step_distance_m",
    "time_delta_s",
    "gps_speed_mps",
    "has_motion_context",
    "candidate_rank",
    "from_node_in_degree",
    "to_node_out_degree"
]

TARGET_COLUMN = "target"


def main():
    print("Loading training dataset...")

    data = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    print(f"Total rows: {len(data):,}")

    # Use a manageable stratified sample for the first benchmark.
    sample = data.sample(
        n=100_000,
        random_state=42
    )

    X = sample[FEATURE_COLUMNS]
    y = sample[TARGET_COLUMN]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"KNN training rows: {len(X_train):,}")
    print(f"KNN validation rows: {len(X_valid):,}")

    model = build_model()

    print("\nTraining KNN...")
    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    print(f"Training completed in {training_time:.2f} seconds.")

    print("\nPredicting validation rows...")
    start = time.time()

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    prediction_time = time.time() - start

    print(f"Prediction completed in {prediction_time:.2f} seconds.")

    print("\nKNN Benchmark Results")
    print("---------------------")
    print(f"Accuracy:   {accuracy_score(y_valid, predictions):.4f}")
    print(f"Precision:  {precision_score(y_valid, predictions, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(y_valid, predictions, zero_division=0):.4f}")
    print(f"F1 score:   {f1_score(y_valid, predictions, zero_division=0):.4f}")
    print(f"Log loss:   {log_loss(y_valid, probabilities):.4f}")
    print(f"Training time:   {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")


if __name__ == "__main__":
    main()
