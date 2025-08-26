"""Model implementations for the resource prediction project.

This package contains production-ready model implementations and interfaces for
the resource prediction project. All models implement the BasePredictor
interface for consistency across training and application layers.
"""

from .base import BasePredictor
from .implementations.lightgbm_models import LightGBMClassifier, LightGBMRegressor
from .implementations.quantile_ensemble_variants import (
    GBLGBQuantileEnsemble,
    GBXGBQuantileEnsemble,
    LGBCatQuantileEnsemble,
    LGBXGBQuantileEnsemble,
    XGBCatQuantileEnsemble,
    XGBXGBQuantileEnsemble,
    LGBLGBQuantileEnsemble,
    CatCatQuantileEnsemble,
    LGBRFQuantileEnsemble,
    XGBRFQuantileEnsemble,
    RFRFQuantileEnsemble,
)
from .implementations.sizey_model import SizeyPredictor
from .implementations.sklearn_models import LogisticRegression, RandomForestClassifier, RandomForestQuantileRegressor
from .implementations.xgboost_models import XGBoostClassifier, XGBoostRegressor
from .unified_wrapper import DeployableModel, load_model

__all__ = [
    "BasePredictor",
    "GBXGBQuantileEnsemble",
    "LGBXGBQuantileEnsemble",
    "GBLGBQuantileEnsemble",
    "XGBCatQuantileEnsemble",
    "LGBCatQuantileEnsemble",
    "XGBXGBQuantileEnsemble",
    "LGBLGBQuantileEnsemble",
    "CatCatQuantileEnsemble",
    "LGBRFQuantileEnsemble",
    "XGBRFQuantileEnsemble",
    "RFRFQuantileEnsemble",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "LightGBMRegressor",
    "LightGBMClassifier",
    "RandomForestClassifier",
    "RandomForestQuantileRegressor",
    "LogisticRegression",
    "SizeyPredictor",
    "DeployableModel",
    "load_model",
]
