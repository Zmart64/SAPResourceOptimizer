"""Base model interface for all prediction models."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    Base interface for all prediction models in the resource prediction project.

    This interface standardizes the model API across training and application layers,
    making it easy to add new models and maintain consistency.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """
        Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target values
            **fit_params: Additional fitting parameters
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {}

    def set_params(self, **params) -> "BasePredictor":
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        return self

    # Shared encoding utility for all predictors
    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical/object features consistently and align columns.

        This central implementation eliminates duplicate _encode methods across models.

        Args:
            X: Input feature DataFrame
            fit: If True, learns and stores the encoded column set. If False, aligns to stored columns

        Returns:
            Encoded and aligned DataFrame of floats
        """
        # Create dummy variables (consistent with prior implementations)
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)

        # Defensive: remove any duplicate columns
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]

        # Ensure numeric dtype
        Xd = Xd.astype(float)

        if fit:
            # Store column names for future alignment
            # Backward-compatible attribute name used by existing code
            self.columns = Xd.columns.tolist()
        else:
            # Align columns to match training data
            if not hasattr(self, "columns") or self.columns is None:
                raise ValueError("Model must be fitted before making predictions")

            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0.0

            # Reorder columns to match training
            Xd = Xd[self.columns]

        return Xd
