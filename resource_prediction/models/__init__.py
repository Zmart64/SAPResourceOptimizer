"""Model implementations for the resource prediction project.

This package contains unified model implementations and interfaces for
the resource prediction project. All models implement the BasePredictor
interface for consistency across training and application layers.
"""

from .base import BasePredictor
from .quantile_ensemble import QuantileEnsemblePredictor, QEPredictor
from .unified_wrapper import UnifiedModelWrapper, load_any_model, convert_legacy_models_to_unified

__all__ = [
    "BasePredictor",
    "QuantileEnsemblePredictor", 
    "QEPredictor",  # Alias for backward compatibility
    "UnifiedModelWrapper",
    "load_any_model",
    "convert_legacy_models_to_unified"
]
