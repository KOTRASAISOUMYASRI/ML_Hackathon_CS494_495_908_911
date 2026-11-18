
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from models.logistic_regression import build_model as build_logistic
from models.decision_tree import build_model as build_tree
from models.knn import build_model as build_knn


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "outputs" / "train" / "candidates.csv"
MODEL_DIR = ROOT / "models"
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

KNN_SAMPLE_SIZE = 100_000


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading full training dataset...")

    data = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    print(f"Total training rows: {len(data):,}")

    models = {
        "Logistic Regression": build_logistic(),
        "Decision Tree": build_tree()
    }

    metadata = {}

    # Train Logistic Regression and Decision Tree on all rows.
    for name, model in models.items():
        print(f"\nTraining {name}...")

        start = time.time()
        model.fit(X, y)
        elapsed = time.time() - start

        filename = name.lower().replace(" ", "_") + ".joblib"
        output_path = MODEL_DIR / filename

        joblib.dump(model, output_path)

        metadata[name] = {
            "file": filename,
            "training_rows": len(X),
            "training_time_seconds": round(elapsed, 3)
        }

        print(f"Training completed in {elapsed:.2f} seconds.")
        print(f"Saved: {output_path}")

    # KNN is trained on a stratified sample for practicality.
    print("\nPreparing KNN training sample...")

    X_knn, _, y_knn, _ = train_test_split(
        X,
        y,
        train_size=KNN_SAMPLE_SIZE,
        random_state=42,
        stratify=y
    )

    knn = build_knn()

    print(f"Training KNN on {len(X_knn):,} rows...")

    start = time.time()
    knn.fit(X_knn, y_knn)
    elapsed = time.time() - start

    knn_path = MODEL_DIR / "knn.joblib"
    joblib.dump(knn, knn_path)

    metadata["KNN"] = {
        "file": "knn.joblib",
        "training_rows": len(X_knn),
        "total_available_training_rows": len(X),
        "sampling": "Stratified random sample",
        "random_state": 42,
        "training_time_seconds": round(elapsed, 3)
    }

    print(f"KNN training completed in {elapsed:.2f} seconds.")
    print(f"Saved: {knn_path}")

    metadata_path = RESULTS_DIR / "final_model_training.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nAll baseline models trained and saved.")
    print(f"Training details saved to: {metadata_path}")

    print("\nModel summary:")
    for name, details in metadata.items():
        print(
            f"{name}: {details['training_rows']:,} rows, "
            f"{details['training_time_seconds']} seconds"
        )


if __name__ == "__main__":
    main()
