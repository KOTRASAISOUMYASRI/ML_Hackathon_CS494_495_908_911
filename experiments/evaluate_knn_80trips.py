
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
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
SAMPLE_SIZE = 100000


def calculate_top1(metadata, y, probabilities):
    evaluation = metadata.copy()
    evaluation["target"] = np.asarray(y)
    evaluation["probability"] = probabilities

    grouped = evaluation.groupby(
        ["trip_id", "observation_id"],
        sort=False
    )

    valid_groups = grouped["target"].sum()
    valid_groups = valid_groups[valid_groups > 0].index

    if len(valid_groups) == 0:
        return 0.0

    valid_data = evaluation.set_index(
        ["trip_id", "observation_id"]
    ).loc[valid_groups].reset_index()

    best_indices = valid_data.groupby(
        ["trip_id", "observation_id"]
    )["probability"].idxmax()

    return valid_data.loc[best_indices, "target"].mean()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading all training candidates...")

    data = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    print(f"Total candidate rows: {len(data):,}")
    print(f"Total trips: {data['trip_id'].nunique()}")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    groups = data["trip_id"]

    metadata = data[
        ["trip_id", "observation_id"]
    ].copy()

    cv = GroupKFold(n_splits=5)

    probabilities = np.zeros(len(data), dtype=float)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):
        print(f"\n========== FOLD {fold}/5 ==========")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        print(f"Training rows before sampling: {len(train_idx):,}")
        print(f"Validation rows: {len(val_idx):,}")
        print(
            "Training trips:",
            groups.iloc[train_idx].nunique()
        )
        print(
            "Validation trips:",
            groups.iloc[val_idx].nunique()
        )

        if len(X_train) < SAMPLE_SIZE:
            raise ValueError(
                "Training fold has fewer than 100,000 rows."
            )

        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=SAMPLE_SIZE,
            stratify=y_train,
            random_state=42
        )

        print(f"KNN training sample: {len(X_sample):,}")

        model = build_model()

        train_start = time.time()
        model.fit(X_sample, y_sample)
        training_time = time.time() - train_start

        prediction_start = time.time()
        fold_probabilities = model.predict_proba(X_val)[:, 1]
        prediction_time = time.time() - prediction_start

        probabilities[val_idx] = fold_probabilities

        fold_predictions = (
            fold_probabilities >= 0.5
        ).astype(int)

        fold_macro_f1 = f1_score(
            y_val,
            fold_predictions,
            average="macro",
            zero_division=0
        )

        fold_top1 = calculate_top1(
            metadata.iloc[val_idx],
            y_val,
            fold_probabilities
        )

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Prediction time: {prediction_time:.2f} seconds")
        print(f"Macro F1: {fold_macro_f1:.4f}")
        print(f"Top-1 Accuracy: {fold_top1:.4f}")

        fold_results.append({
            "fold": fold,
            "training_rows": len(X_sample),
            "validation_rows": len(val_idx),
            "macro_f1": fold_macro_f1,
            "top1_accuracy": fold_top1,
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time
        })

        del model

    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "model": "KNN",
        "total_trips": data["trip_id"].nunique(),
        "folds": 5,
        "training_sample_per_fold": SAMPLE_SIZE,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "log_loss": log_loss(y, probabilities),
        "top1_accuracy": calculate_top1(
            metadata, y, probabilities
        )
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(
        RESULTS_DIR / "knn_80trips_results.csv",
        index=False
    )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        RESULTS_DIR / "knn_80trips_fold_results.csv",
        index=False
    )

    print("\n========== FINAL KNN RESULTS ==========")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nFold-wise results:")
    print(fold_df.to_string(index=False))

    print("\nResults saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
