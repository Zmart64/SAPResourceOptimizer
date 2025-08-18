"""Top-level package for the resource prediction project.

This package provides modules for preprocessing build telemetry,
training machine-learning models, and orchestrating the entire
pipeline. The subpackages expose cohesive pieces of functionality:

``data_processing``
    Utilities for transforming raw build logs into model-ready
    features.

``training``
    Hyper-parameter optimisation and evaluation utilities.

``models``
    Placeholder for model implementations. New approaches can be
    added here and wired into the configuration.
"""

from .config import Config
from .data_processing import DataPreprocessor
from .training import Trainer, OptunaOptimizer

__all__ = [
    "Config",
    "DataPreprocessor",
    "Trainer",
    "OptunaOptimizer",
]
