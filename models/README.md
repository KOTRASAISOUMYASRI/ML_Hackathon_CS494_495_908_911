# Candidate classifiers

All five canonical model definitions live together in this directory:

| Model | Module | Factory |
|---|---|---|
| Logistic Regression | logistic_regression.py | build_model |
| KNN | knn.py | build_model |
| Decision Tree | decision_tree.py | build_model |
| Random Forest | random_forest.py | create_random_forest |
| HistGradientBoosting | hist_gradient_boosting.py | create_hist_gradient_boosting |

The package also exports all five constructors with `create_...` names.
Logistic Regression, Decision Tree and KNN joblib files remain in this directory.
The merged Random Forest and HistGradientBoosting baselines are in
`../outputs/models/`; optimized variants are in `../outputs/optimization/`.
Run `git lfs pull` to download the actual large files after cloning.
The route-recovery registry discovers these locations without moving artifacts
or modifying the training definitions. See the
[measured integration comparison](../reports/integrated_models/SUMMARY.md).

Training and evaluation scripts live in `../experiments/`. From the repository
root, run the relocated advanced training modules with:

```sh
python -m experiments.train_random_forest
python -m experiments.train_hist_gradient_boosting
```

Those scripts retain Member 3's dataset paths and output locations. Check them
before training. Bayesian optimisation remains under
`../src/pathsnap/optimization/`. Old `pathsnap.models` imports are thin
compatibility shims pointing here; there are no duplicate model definitions.
