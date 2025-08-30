""" "Random forest predictor."""

# Code from jonathanbader
# Source: https://github.com/dos-group/sizey

from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from resource_prediction.models.implementations.sizey import (
    PredictionModel,
)

simplefilter("ignore", category=ConvergenceWarning)


class RandomForestPredictor(PredictionModel):
    """Random Forest Regression Predictor"""

    def __init__(
        self, workflow_name: str, task_name: str, err_metr: str, random_state: int
    ):
        super().__init__(workflow_name, task_name, err_metr)
        self.random_state: int = random_state
        self.x_train_full: np.ndarray = None
        self.y_train_full: np.ndarray = None
        self.model_error: float = None
        self.regressor: RandomForestRegressor = None

    def initial_model_training(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Initial model training."""
        # Initialize internal storage of historical values
        self.x_train_full = x_train
        self.y_train_full = y_train

        self._search_best_model(x_train, y_train)

    def predict_task(self, task_features: np.ndarray) -> float:
        """Predict a single task."""
        task_features_scaled = self.train_x_scaler.transform(task_features)

        return self.train_y_scaler.inverse_transform(
            self.regressor.predict(task_features_scaled).reshape(-1, 1)
        )

    def predict_tasks(self, tasks_dataframe: np.ndarray) -> np.ndarray:
        """Predict multiple tasks."""
        tasks_dataframe_scaled = self.train_x_scaler.transform(tasks_dataframe)

        return self.train_y_scaler.inverse_transform(
            self.regressor.predict(tasks_dataframe_scaled).reshape(-1, 1)
        )

    def update_model(self, x_train: pd.Series, y_train: float) -> None:
        """Update the model with new data."""
        # Append the newly incoming data to maintain all historical data

        self.x_train_full = np.concatenate(
            (self.x_train_full, x_train.values.reshape(1, -1))
        )
        self.y_train_full = np.concatenate(
            (self.y_train_full, np.array([y_train]).reshape(-1, 1))
        )

        # Scaling of data with all historical data
        self.train_x_scaler = self.train_x_scaler.fit(self.x_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Retrain existing model with scaled data
        self._search_best_model(self.x_train_full, self.y_train_full)

    def smoothed_mape(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
    ) -> float:
        """Compute the smoothed Mean Absolute Percentage Error (MAPE)."""
        # y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Calculate the individual percentage errors and clip each at 100%
        mape = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = -1 * np.clip(mape, None, 1)  # Clip values at 100%
        return np.mean(mape)

    def _search_best_model(self, x_train: np.ndarray, y_train: np.ndarray):
        """Grid search to find best model for training data"""
        self.train_x_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        x_train_scaled = self.train_x_scaler.fit_transform(x_train)
        y_train_scaled = self.train_y_scaler.fit_transform(y_train)

        smoothed_mape_scorer = make_scorer(self.smoothed_mape, greater_is_better=True)

        # Define parameter grid for RandomForestRegressor
        param_grid = {
            "n_estimators": [10, 25, 50],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "max_features": [1],
        }

        # Create RandomForestRegressor model
        model = RandomForestRegressor(random_state=42)

        if self.err_metr == "smoothed_mape":
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                error_score="raise",
                n_jobs=-1,
                scoring=smoothed_mape_scorer,
            )
        elif self.err_metr == "neg_mean_squared_error":
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                error_score="raise",
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
        else:
            raise NotImplementedError("Error metric not found.")

        grid_search.fit(x_train_scaled, y_train_scaled.ravel())

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model_error = best_score
        self.regressor = best_model
