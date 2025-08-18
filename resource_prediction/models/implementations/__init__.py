"""Specific model implementations for the resource prediction project.

This module contains all the concrete model implementations organized by
framework/library (LightGBM, XGBoost, scikit-learn, etc.).
"""

from .quantile_ensemble import QuantileEnsemblePredictor
from .xgboost_models import XGBoostRegressor, XGBoostClassifier
from .lightgbm_models import LightGBMRegressor, LightGBMClassifier
from .sklearn_models import RandomForestClassifier, LogisticRegression

__all__ = [
    "QuantileEnsemblePredictor",
    "XGBoostRegressor", 
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier", 
    "RandomForestClassifier",
    "LogisticRegression"
]
