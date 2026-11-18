"""Compatibility exports; canonical model definitions live in top-level models."""
from models.random_forest import create_random_forest
from models.hist_gradient_boosting import create_hist_gradient_boosting

__all__ = [
    "create_random_forest",
    "create_hist_gradient_boosting",
]
