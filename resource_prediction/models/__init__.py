"""Model implementations for the resource prediction project.

This package contains production-ready model implementations and interfaces for
the resource prediction project. All models implement the BasePredictor
interface for consistency across training and application layers.
"""

from .base import BasePredictor
from .implementations.quantile_ensemble import QuantileEnsemblePredictor
from .implementations.xgboost_models import XGBoostRegressor, XGBoostClassifier
from .implementations.lightgbm_models import LightGBMRegressor, LightGBMClassifier
from .implementations.sklearn_models import RandomForestClassifier, LogisticRegression
from .unified_wrapper import DeployableModel, load_model

__all__ = [
    "BasePredictor",
    "QuantileEnsemblePredictor",
    "XGBoostRegressor", 
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier", 
    "RandomForestClassifier",
    "LogisticRegression",
    "DeployableModel",
    "load_model"
]
