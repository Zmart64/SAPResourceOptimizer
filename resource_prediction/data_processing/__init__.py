"""Data preprocessing utilities for the resource prediction project.

The :mod:`data_processing` package transforms raw build telemetry into
features suitable for machine-learning models. It currently exposes a
:class:`~resource_prediction.data_processing.preprocessor.DataPreprocessor`
class that performs extensive feature engineering and train/test
splitting.
"""

from .preprocessor import DataPreprocessor

__all__ = ["DataPreprocessor"]
