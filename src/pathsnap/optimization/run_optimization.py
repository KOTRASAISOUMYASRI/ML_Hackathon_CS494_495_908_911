
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

print("Loading training data...", flush=True)

df = pd.read_csv(DATA, dtype={"trip_id": str})

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]
groups = df["trip_id"]

print("Training rows:", len(df), flush=True)
print("Unique trips:", groups.nunique(), flush=True)

print("\nStarting Random Forest optimisation...", flush=True)

start = time.time()

best_model, study = optimize_model(
    model_name="random_forest",
    X=X,
    y=y,
    groups=groups,
    n_trials=2,
    random_state=42,
)

print("\nOptimisation completed!", flush=True)
print("Best CV F1:", study.best_value, flush=True)
print("Best parameters:", study.best_params, flush=True)
print("Total time:", round(time.time() - start, 2), "seconds", flush=True)

joblib.dump(
    best_model,
    OUTPUT / "random_forest_optimized.joblib",
)

study.trials_dataframe().to_csv(
    OUTPUT / "random_forest_trials.csv",
    index=False,
)

print("Optimised model and trial results saved.", flush=True)
