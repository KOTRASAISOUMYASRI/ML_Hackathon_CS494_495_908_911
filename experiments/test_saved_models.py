
from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    ConfusionMatrixDisplay,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
TEST_DIR = ROOT / "outputs" / "student_test"
TEST_PATH = TEST_DIR / "candidates.csv"
OBSERVATIONS_PATH = TEST_DIR / "observations.csv"
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


def calculate_top1(data, probabilities):
    evaluation_data = data[
        ["trip_id", "observation_id", "target"]
    ].copy()

    evaluation_data["probability"] = probabilities

    correct = 0
    eligible = 0

    for _, group in evaluation_data.groupby(
        ["trip_id", "observation_id"]
    ):
        # Exclude observations where no candidate is correct.
        if group["target"].sum() == 0:
            continue

        eligible += 1

        # Select the candidate with the highest probability.
        best_candidate = group.loc[
            group["probability"].idxmax()
        ]

        if best_candidate["target"] == 1:
            correct += 1

    if eligible == 0:
        return 0.0

    return correct / eligible


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading student-test dataset...")

    data = pd.read_csv(
        TEST_PATH,
        dtype={"trip_id": str}
    )

    observations = pd.read_csv(
        OBSERVATIONS_PATH,
        dtype={"trip_id": str}
    )

    X_test = data[FEATURE_COLUMNS]
    y_test = data["target"]

    print(f"Student-test candidate rows: {len(data):,}")

    eligible_observations = (
        data.groupby(["trip_id", "observation_id"])["target"]
        .sum()
        .gt(0)
        .sum()
    )

    total_observations = len(observations)
    candidate_misses = int(
        observations["candidate_miss"].astype(bool).sum()
    )

    print(f"Total GPS observations: {total_observations:,}")
    print(f"Observations with a candidate match: {eligible_observations:,}")
    print(f"Candidate misses: {candidate_misses:,}")
    print(
        f"Candidate miss rate: "
        f"{candidate_misses / total_observations:.2%}"
    )

    model_files = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "knn.joblib"
    ]

    results = []

    for filename in model_files:
        model_name = filename.replace(".joblib", "")

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(MODEL_DIR / filename)

        start_time = time.time()
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time

        top1 = calculate_top1(data, probabilities)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "log_loss": log_loss(y_test, probabilities),
            "top1_accuracy": top1,
            "prediction_time_seconds": prediction_time
        }

        results.append(metrics)

        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

        matrix = confusion_matrix(y_test, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Negative", "Positive"]
        )

        display.plot()
        plt.title(f"{model_name} - Student Test Confusion Matrix")
        plt.tight_layout()

        image_path = (
            RESULTS_DIR /
            f"{model_name}_student_test_confusion_matrix.png"
        )

        plt.savefig(image_path, dpi=150)
        plt.close()

        print(f"Saved confusion matrix: {image_path.name}")

    results_df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "student_test_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    print("\nStudent-test comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved results: {csv_path}")


if __name__ == "__main__":
    main()
