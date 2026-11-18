
from pathlib import Path
import sys
import time
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from pathsnap.optimization.optimize import optimize_model

DATA = ROOT / "Processed Dataset" / "outputs" / "train" / "candidates.csv"
OUTPUT = ROOT / "outputs" / "optimization"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SEARCH_TRIPS = 20
N_TRIALS = 4
N_SPLITS = 3

print("Loading training data...", flush=True)
df = pd.read_csv(DATA, dtype={"trip_id": str})

print("Training rows:", len(df), flush=True)
print("Total trips:", df["trip_id"].nunique(), flush=True)

# Select complete trips, not random candidate rows.
all_trips = sorted(df["trip_id"].unique())
rng = pd.Series(all_trips).sample(
    n=min(SEARCH_TRIPS, len(all_trips)),
    random_state=RANDOM_STATE,
).tolist()

search_df = df[df["trip_id"].isin(rng)].copy()

X_search = search_df[FEATURE_COLUMNS]
y_search = search_df[TARGET_COLUMN]
groups_search = search_df["trip_id"]

X_full = df[FEATURE_COLUMNS]
y_full = df[TARGET_COLUMN]

print("Search trips:", len(rng), flush=True)
print("Search rows:", len(search_df), flush=True)
print("Final refit rows:", len(df), flush=True)

models = ["random_forest", "hist_gradient_boosting"]

for model_name in models:
    print(f"\nStarting {model_name} optimization...", flush=True)
    start = time.time()

    best_model, study = optimize_model(
        model_name=model_name,
        X=X_search,
        y=y_search,
        groups=groups_search,
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        refit_X=X_full,
        refit_y=y_full,
        random_state=RANDOM_STATE,
    )

    joblib.dump(
        best_model,
        OUTPUT / f"{model_name}_optimized.joblib",
    )

    study.trials_dataframe().to_csv(
        OUTPUT / f"{model_name}_trials.csv",
        index=False,
    )

    print(f"{model_name} completed!", flush=True)
    print("Best CV F1:", study.best_value, flush=True)
    print("Best parameters:", study.best_params, flush=True)
    print("Time:", round(time.time() - start, 2), "seconds", flush=True)

print("\nBoth optimizations completed.", flush=True)
print("Results saved in:", OUTPUT, flush=True)
