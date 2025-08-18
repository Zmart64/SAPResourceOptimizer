"""
SizeyPredictor - A wrapper around Sizey for resource prediction.
Note: Sizey is supposed to be used for online learning.
In this wrapper, we use heuristic dummy values for y_test.
"""

import numpy as np

from ...base import BasePredictor
from .experiment_constants import ERROR_STRATEGY, OFFSET_STRATEGY
from .sizing_tasks import Sizey


class SizeyPredictor(BasePredictor):
    """
    Sizey is an ensemble-based machine learning approach.
    Ensemble comprises of:
        - Linear Regression
        - Neural Network
        - Random Forest
        - KNN

    Sizey uses Resource Allocation Quality (RAQ)
    to select the best predictor or ensemble predictions.
    """

    def __init__(
        self,
        sizey_alpha: float = 0.1,
        offset_strat: str = "DYNAMIC",
        error_strat: str = "MAX_EVER_OBSERVED",
        use_softmax: bool = True,
        error_metric: str = "smoothed_mape",
    ):
        """
        Initialize the Sizey Predictor.

        Args:
            sizey_alpha: Set the alpha. It has to be between 0.0 and 1.0
            offset_strat: Offset strategy (DYNAMIC, STD, MED_UNDER, MED_ALL, STDUNDER)
            error_strat: Error strategy (MAX_EVER_OBSERVED, DOUBLE)
            use_softmax: Interpolation strategy, softmax if True, else argmax
            error_metric: smoothed_mape or neg_mean_squared_error
        """
        self.sizey_alpha = sizey_alpha
        self.offset_strat = offset_strat
        self.error_strat = error_strat
        self.use_softmax = use_softmax
        self.error_metric = error_metric
        self.sizey_model = None
        self.is_fitted = False

        # Convert string parameters to enum values
        self._offset_strategy = self._get_offset_strategy(offset_strat)
        self._error_strategy = self._get_error_strategy(error_strat)

    def _get_offset_strategy(self, offset_strat: str):
        """Convert string offset strategy to enum."""

        strategy_map = {
            "DYNAMIC": OFFSET_STRATEGY.DYNAMIC,
            "STD": OFFSET_STRATEGY.STD,
            "MED_UNDER": OFFSET_STRATEGY.MED_UNDER,
            "MED_ALL": OFFSET_STRATEGY.MED_ALL,
            "STDUNDER": OFFSET_STRATEGY.STDUNDER,
        }
        return strategy_map.get(offset_strat, OFFSET_STRATEGY.DYNAMIC)

    def _get_error_strategy(self, error_strat: str):
        """Convert string error strategy to enum."""

        strategy_map = {
            "MAX_EVER_OBSERVED": ERROR_STRATEGY.MAX_EVER_OBSERVED,
            "DOUBLE": ERROR_STRATEGY.DOUBLE,
        }
        return strategy_map.get(error_strat, ERROR_STRATEGY.MAX_EVER_OBSERVED)

    def fit(self, X, y, **fit_params):
        """
        Fit the Sizey model to training data.

        Args:
            X: Feature matrix
            y: Target values
            **fit_params: Additional fitting parameters
        """

        # Convert y to numpy array and reshape to 2D as expected by Sizey
        # Debug shapes before and after reshaping
        y_array = y.values.reshape(-1, 1)
        print(f"Reshaped y_array shape: {y_array.shape}")

        # Initialize the Sizey model with training data
        self.sizey_model = Sizey(
            X_train=X,
            y_train=y_array,
            alpha=self.sizey_alpha,
            offset_strategy=self._offset_strategy,
            default_offset=0.1,  # Default offset value
            error_strategy=self._error_strategy,
            use_softmax=self.use_softmax,
            error_metric=self.error_metric,
        )

        self.is_fitted = True

    def predict(self, X, y=None):
        """
        Make predictions using the Sizey model.

        Args:
            X: Feature matrix (pandas DataFrame)
            y: True target value (pandas Series)

        Returns:
            Predicted values (numpy array)
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        predictions = []

        for idx in range(len(X)):
            # Get the row as a pandas Series
            x_test = X.iloc[idx]
            y_test = y.iloc[idx]

            # Get prediction from Sizey model
            # Sizey returns (prediction_with_offset, raw_prediction)
            prediction_with_offset, raw_prediction = self.sizey_model.predict(
                X_test=x_test, y_test=y_test, user_estimate=None
            )

            # Extract scalar value from prediction_with_offset (may be a nested array)
            if hasattr(prediction_with_offset, "flatten"):
                prediction_value = prediction_with_offset.flatten()[0]
            else:
                prediction_value = prediction_with_offset

            print(f"Raw prediction: {raw_prediction}")

            predictions.append(prediction_value)

        return np.array(predictions)
