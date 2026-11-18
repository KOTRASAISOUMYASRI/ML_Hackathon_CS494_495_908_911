
from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

from models.logistic_regression import build_model as build_logistic
from models.decision_tree import build_model as build_tree
from models.knn import build_model as build_knn


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TRAIN_PATH = ROOT / "outputs" / "train" / "candidates.csv"

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


def top1_accuracy(metadata, y, probabilities):
    evaluation = metadata.copy()
    evaluation["target"] = y.to_numpy()
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

    best = valid_data.groupby(
        ["trip_id", "observation_id"]
    )["probability"].idxmax()

    return valid_data.loc[best, "target"].mean()


def evaluate(name, model, X, y, groups, metadata):
    print(f"\nEvaluating {name}...")
    start = time.time()

    cv = GroupKFold(n_splits=5)

    probabilities = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    elapsed = time.time() - start

    result = {
        "model": name,
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
        "log_loss": log_loss(y, probabilities),
        "top1_accuracy": top1_accuracy(
            metadata, y, probabilities
        ),
        "time_seconds": elapsed
    }

    for metric, value in result.items():
        if metric not in ["model", "time_seconds"]:
            print(f"{metric}: {value:.4f}")

    print(f"Time: {elapsed:.2f} seconds")

    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading training data...")

    data = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    # Select five complete trips, not random candidate rows.
    trip_ids = sorted(data["trip_id"].unique())

    selected_trips = pd.Series(trip_ids).sample(
        n=5,
        random_state=42
    ).tolist()

    data = data[data["trip_id"].isin(selected_trips)].copy()

    print("\nSelected trips:")
    print(selected_trips)

    print(f"\nRows selected: {len(data):,}")
    print(f"Number of trips: {data['trip_id'].nunique()}")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    groups = data["trip_id"]

    metadata = data[
        ["trip_id", "observation_id"]
    ].copy()

    models = {
        "Logistic Regression": build_logistic(),
        "Decision Tree": build_tree(),
        "KNN": build_knn()
    }

    results = []

    for name, model in models.items():
        result = evaluate(
            name, model, X, y, groups, metadata
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "baseline_comparison_grouped_sample.csv"
    results_df.to_csv(output_path, index=False)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "top1_accuracy"
    ]

    results_df.set_index("model")[metrics].plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Baseline Model Comparison - Grouped Sample")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "baseline_comparison_grouped_sample.png"
    )
    plt.close()

    print("\nFinal comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
