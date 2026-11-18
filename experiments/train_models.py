
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "outputs" / "train" / "candidates.csv"
TEST_PATH = ROOT / "outputs" / "student_test" / "candidates.csv"

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

ID_COLUMNS = [
    "trip_id",
    "observation_id",
    "candidate_arc_id"
]


def load_datasets():
    print("Loading training dataset...")

    train = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    print("Loading student-test dataset...")

    test = pd.read_csv(
        TEST_PATH,
        dtype={"trip_id": str}
    )

    required_columns = (
        FEATURE_COLUMNS + [TARGET_COLUMN] + ID_COLUMNS
    )

    for name, dataset in [("train", train), ("test", test)]:
        missing = set(required_columns) - set(dataset.columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {missing}"
            )

        if dataset[FEATURE_COLUMNS].isna().any().any():
            raise ValueError(f"{name} has missing feature values.")

        if not np.isfinite(
            dataset[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(f"{name} has non-finite feature values.")

        if not set(dataset[TARGET_COLUMN].unique()).issubset({0, 1}):
            raise ValueError(f"{name} has invalid target labels.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    groups = train["trip_id"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    print("\nDataset loaded successfully.")
    print(f"Training rows: {len(train):,}")
    print(f"Student-test rows: {len(test):,}")
    print(f"Training trips: {groups.nunique()}")
    print(f"Student-test trips: {test['trip_id'].nunique()}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True) * 100)

    return X_train, y_train, groups, X_test, y_test, test


if __name__ == "__main__":
    load_datasets()
