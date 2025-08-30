"""
SizeyPredictor - A wrapper around Sizey for resource prediction.
"""

import numpy as np
import pandas as pd

from resource_prediction.models.base import BasePredictor
from resource_prediction.models.implementations.sizey import (
    OffsetStrategy,
    Sizey,
    UnderPredictionStrategy,
)


class SizeyPredictor(BasePredictor):
    """
    Wrapper around the sizey ensemble.
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
        alpha: float = 0.1,
        beta: float = 1.0,
        offset_strat: OffsetStrategy = OffsetStrategy.DYNAMIC,
        error_strat: UnderPredictionStrategy = UnderPredictionStrategy.MAX_EVER_OBSERVED,
        use_softmax: bool = True,
        error_metric: str = "smoothed_mape",
        random_state: int = 42,
    ):
        """
        Initialize the Sizey Predictor.
        Args:
            alpha (float): Alpha parameter for Sizey model (0.0 to 1.0)
            offset_strat (OffsetStrategy): Offset strategy for predictions
            error_strat (UnderPredictionStrategy): Error strategy for model selection
            use_softmax (bool): Whether to use softmax for interpolation
            error_metric (str): Error metric to use
        """

        self.alpha = alpha
        self.beta = beta
        self.offset_strat = offset_strat
        self.error_strat = error_strat
        self.use_softmax = use_softmax
        self.error_metric = error_metric
        self.random_state = random_state
        self.sizey_model = None
        self.is_fitted = False
        self.columns = None
        self.original_columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical features and align columns.

        Args:
            X: Input DataFrame
            fit: Whether to fit the encoder (store column names)

        Returns:
            Encoded DataFrame with aligned columns
        """
        # Create dummy variables
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)

        # Handle duplicate columns (shouldn't happen but defensive programming)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]

        if fit:
            # Store column names for future alignment
            self.columns = Xd.columns.tolist()
        else:
            # Align columns to match training data
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")

            # Add missing columns with zeros
            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0

            # Reorder columns to match training
            Xd = Xd[self.columns]

        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """
        Fit the Sizey model to training data.

        Args:
            X: Feature matrix
            y: Target values
            **fit_params: Additional fitting parameters
        """
        X.to_csv("debug_X_fit.csv", index=False)

        # Encode features
        x_encoded = self._encode(X, fit=True)
        x_encoded.to_csv("debug_X_fit_encoded.csv", index=False)

        # Convert x_encoded to numpy array
        x_array = x_encoded.to_numpy()

        # Convert y to numpy array and reshape to 2D as expected by Sizey
        # Debug shapes before and after reshaping
        y_array = y.values.reshape(-1, 1)

        # Initialize the Sizey model with training data
        self.sizey_model = Sizey(
            x_train=x_array,
            y_train=y_array,
            alpha=self.alpha,
            beta=self.beta,
            offset_strategy=self.offset_strat,
            default_offset=0.1,
            error_strategy=self.error_strat,
            use_softmax=self.use_softmax,
            error_metric=self.error_metric,
            random_state=self.random_state,
        )

        self.is_fitted = True

    def predict(self, X: pd.DataFrame, y: pd.Series):
        """
        Make prediction using the Sizey model.
        IMPORTANT: After each prediction the model expects error calculation
        Therefore calculate_error should be called afterwards

        Args:
            X: Feature matrix

        Returns:
            Predicted value (numpy array)
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        x_encoded = self._encode(X, fit=False)
        x_array = x_encoded.to_numpy()

        for i in range(len(y)):
            y_instance = y.iloc[i]
            x_test_instance = x_array[i].reshape(1, -1)
            # Get prediction from Sizey model
            # Sizey returns (scaled_prediction (with offset), raw_prediction)
            scaled_prediction, _ = self.sizey_model.predict(x_test=x_test_instance)
            self.sizey_model.calculate_error(y_instance)

        if isinstance(scaled_prediction, np.ndarray):
            scaled_prediction = scaled_prediction.flatten()[0]

        return scaled_prediction

        # ATTENTION
        # Normally sizey does online learning, meaning:
        # 1. underpredictions are handled
        # 2. the model is updated with new data

        # We are not doing this, to compare against other models

        # Handle underpredictions according to sizey
        # if scaled_prediction < y_test:
        #     # Apply sizey-specific underprediction handling
        #     scaled_prediction = self.sizey_model.handle_underprediction(
        #         predicted=scaled_prediction
        #     )

        # Update the model with the new data point
        # self.sizey_model.update_model(x_test, y_test)

    def calculate_error(self, y_true) -> None:
        """
        Calculate the error of the model's predictions.

        Args:
            y_true: True target values

        Returns:
            None
        """

        self.sizey_model.calculate_error(y_true)
