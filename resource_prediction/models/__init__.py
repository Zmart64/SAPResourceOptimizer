"""Model implementations for the resource prediction project.

This package contains production-ready model implementations and interfaces for
the resource prediction project. All models implement the BasePredictor
interface for consistency across training and application layers.
"""

from .base import BasePredictor
from .quantile_ensemble import QuantileEnsemblePredictor
from .unified_wrapper import DeployableModel, load_model

__all__ = [
    "BasePredictor",
    "QuantileEnsemblePredictor", 
    "DeployableModel",
    "load_model"
]
