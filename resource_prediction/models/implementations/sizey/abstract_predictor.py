"""Abstract base class for all predictors."""

# Code from jonathanbader
# Source: https://github.com/dos-group/sizey

from abc import ABCMeta

import numpy as np
import pandas as pd


class PredictionModel(metaclass=ABCMeta):
    """
    Abstract base class for prediction models used in workflow task resource prediction.

    This class defines the interface for prediction models that can be trained on historical
    task execution data and used to predict resource requirements for future tasks.

    Attributes:
        workflow_name (str): Name of the workflow this model is associated with.
        task_name (str): Name of the specific task this model predicts.
        err_metr (str): Error metric used for model evaluation.
        regressor: The underlying regression model (implementation-specific).
        train_x_scaler: Scaler for input features normalization.
        train_y_scaler: Scaler for target values normalization.
        x_train_full: Full training dataset features.
        y_train_full: Full training dataset targets.
        model_error: Current model error metric value.

    Methods:
        initial_model_training: Train the model on initial dataset.
        predict_task: Predict resource requirements for a single task.
        predict_tasks: Predict resource requirements for multiple tasks.
        update_model: Update the model with new training data.
    """

    def __init__(self, workflow_name: str, task_name: str, err_metr: str):
        self.workflow_name = workflow_name
        self.task_name = task_name
        self.err_metr = err_metr
        self.regressor = None
        self.train_x_scaler = None
        self.train_y_scaler = None
        self.x_train_full = None
        self.y_train_full = None
        self.model_error = None

    def initial_model_training(self, x_train, y_train) -> None:
        """Train the model on initial dataset."""
        raise NotImplementedError("Model prediction method has not been implemented.")

    def predict_task(self, task_features: pd.Series) -> np.ndarray:
        """Predict resource requirements for a single task."""
        raise NotImplementedError("Model prediction method has not been implemented.")

    def predict_tasks(self, tasks_dataframe: pd.DataFrame) -> float:
        """Predict resource requirements for multiple tasks."""
        raise NotImplementedError("Predicting multiple tasks has not been implemented.")

    def update_model(self, x_train: pd.Series, y_train: float) -> None:
        """Update the model with new training data."""
        raise NotImplementedError("Model update method has not been implemented.")
