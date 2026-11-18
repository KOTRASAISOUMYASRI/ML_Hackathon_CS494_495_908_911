
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from models.logistic_regression import build_model as build_logistic_model
from models.decision_tree import build_model as build_decision_tree


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TRAIN_PATH = ROOT / "outputs" / "train" / "candidates.csv"
OBSERVATIONS_PATH = ROOT / "outputs" / "train" / "observations.csv"

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


def top1_accuracy(data):
    """
    For each observation, select the candidate with the highest
    predicted probability. Evaluate only observations that have
    at least one positive candidate.
    """
    grouped = data.groupby(
        ["trip_id", "observation_id"],
        sort=False
    )

    valid_groups = grouped[TARGET_COLUMN].sum()
    valid_groups = valid_groups[valid_groups > 0].index

    valid_data = data.set_index(
        ["trip_id", "observation_id"]
    ).loc[valid_groups].reset_index()

    best_indices = valid_data.groupby(
        ["trip_id", "observation_id"]
    )["predicted_probability"].idxmax()

    selected = valid_data.loc[best_indices]

    return selected[TARGET_COLUMN].mean()


def evaluate_model(name, model, X, y, groups, metadata):
    print(f"\nEvaluating {name}...")
    start_time = time.time()

    cv = GroupKFold(n_splits=5)

    probabilities = cross_val_predict(
        clone(model),
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    elapsed = time.time() - start_time

    predictions = (probabilities >= 0.5).astype(int)

    results = {
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
        "training_and_prediction_time_seconds": elapsed
    }

    evaluation_data = metadata.copy()
    evaluation_data[TARGET_COLUMN] = y.to_numpy()
    evaluation_data["predicted_probability"] = probabilities

    results["top1_accuracy"] = top1_accuracy(evaluation_data)

    print(f"Accuracy:   {results['accuracy']:.4f}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"F1 score:   {results['f1']:.4f}")
    print(f"Log loss:   {results['log_loss']:.4f}")
    print(f"Top-1:      {results['top1_accuracy']:.4f}")
    print(f"Time:       {elapsed:.2f} seconds")

    matrix = confusion_matrix(y, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Incorrect candidate", "Correct candidate"]
    )

    display.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.close()

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading training candidates...")
    data = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    groups = data["trip_id"]

    metadata = data[
        ["trip_id", "observation_id"]
    ].copy()

    print("\nStarting five-fold GroupKFold evaluation.")
    print("All observations from a trip stay in the same fold.")

    models = {
        "Logistic Regression": build_logistic_model(),
        "Decision Tree": build_decision_tree()
    }

    all_results = []

    for name, model in models.items():
        result = evaluate_model(
            name, model, X, y, groups, metadata
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\nCandidate coverage information:")
    observations = pd.read_csv(
        OBSERVATIONS_PATH,
        dtype={"trip_id": str}
    )

    misses = int(observations["candidate_miss"].sum())
    total = len(observations)

    print(f"Total observations: {total:,}")
    print(f"Candidate misses: {misses:,}")
    print(f"Candidate miss rate: {misses / total:.2%}")

    metrics = ["accuracy", "precision", "recall", "f1", "top1_accuracy"]

    results_df.set_index("model")[metrics].plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png")
    plt.close()

    print("\nEvaluation completed.")
    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
