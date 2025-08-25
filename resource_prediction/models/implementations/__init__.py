"""Specific model implementations for the resource prediction project.

This module contains all the concrete model implementations organized by
framework/library (LightGBM, XGBoost, scikit-learn, etc.).
"""

from .lightgbm_models import LightGBMClassifier, LightGBMRegressor
from .quantile_ensemble_variants import (
    GBLGBQuantileEnsemble,
    GBXGBQuantileEnsemble,
    LGBCatQuantileEnsemble,
    LGBXGBQuantileEnsemble,
    XGBCatQuantileEnsemble,
)
from .sizey_model import SizeyPredictor
from .sklearn_models import LogisticRegression, RandomForestClassifier
from .xgboost_models import XGBoostClassifier, XGBoostRegressor

__all__ = [
    "GBXGBQuantileEnsemble",
    "LGBXGBQuantileEnsemble",
    "GBLGBQuantileEnsemble",
    "XGBCatQuantileEnsemble",
    "LGBCatQuantileEnsemble",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "SizeyPredictor",
]
