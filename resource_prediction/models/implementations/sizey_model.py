"""
SizeyPredictor - A wrapper around Sizey for resource prediction.
"""

import numpy as np

from resource_prediction.models.base import BasePredictor
from resource_prediction.models.implementations.sizey import (
    OffsetStrategy,
    Sizey,
    UnderPredictionStrategy,
)


class SizeyPredictor(BasePredictor):
    """
    Wrapper around the Sizey ensemble.

    Ensemble comprises of: Linear Regression, Neural Network, Random Forest, and KNN.

    Sizey uses Resource Allocation Quality (RAQ) to select the best predictor or
    combine predictions conservatively for allocation.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        offset_strat: OffsetStrategy = OffsetStrategy.DYNAMIC,
        error_strat: UnderPredictionStrategy = UnderPredictionStrategy.MAX_EVER_OBSERVED,
        use_softmax: bool = True,
        error_metric: str = "smoothed_mape",
    ):
        """
        Initialize the Sizey Predictor.
        Args:
            alpha (float): Alpha parameter for Sizey model (0.0 to 1.0)
            offset_strat (OffsetStrategy): Offset strategy for predictions
            error_strat (ErrorStrategy): Error strategy for model selection
            use_softmax (bool): Whether to use softmax for interpolation
            error_metric (str): Error metric to use
        """
        self.alpha = alpha
        self.beta = beta
        self.offset_strat = offset_strat
        self.error_strat = error_strat
        self.use_softmax = use_softmax
        self.error_metric = error_metric
        self.sizey_model = None
        self.is_fitted = False

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
            x_train=X,
            y_train=y_array,
            alpha=self.alpha,
            beta=self.beta,
            offset_strategy=self.offset_strat,
            default_offset=0.1,
            error_strategy=self.error_strat,
            use_softmax=self.use_softmax,
            error_metric=self.error_metric,
        )

        self.is_fitted = True

    def predict(self, X, y=None):
        """
        Make predictions using the Sizey model.

        Args:
            X: Feature matrix
            y: True target value

        Note:
            Because Sizey is online learning we pass the target value

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
            # Sizey returns (scaled_prediction (with offset), raw_prediction)
            scaled_prediction, _ = self.sizey_model.predict(
                x_test=x_test, y_test=y_test
            )

            if isinstance(scaled_prediction, np.ndarray):
                scaled_prediction = scaled_prediction.flatten()[0]
            predictions.append(scaled_prediction)

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

        print("Finished predictions")

        return np.array(predictions)
