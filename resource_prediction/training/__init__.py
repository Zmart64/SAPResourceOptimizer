"""Training and evaluation utilities for resource prediction models.

The :mod:`training` package contains modules that handle hyper-parameter
optimisation with Optuna and final model evaluation.  It provides:

``hyperparameter``
    Functions and classes to perform Optuna searches across multiple
    model families.

``trainer``
    The high-level orchestration logic that ties preprocessing,
    optimisation, and reporting together.
"""

from .hyperparameter import OptunaOptimizer
from .trainer import Trainer

__all__ = ["OptunaOptimizer", "Trainer"]
