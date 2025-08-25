"""Abstract base class for all predictors."""

# Code from jonathanbader
# Source: https://github.com/dos-group/sizey

from abc import ABCMeta

import numpy as np
import pandas as pd


class PredictionModel(metaclass=ABCMeta):
    def __init__(self, workflow_name: str, task_name: str, err_metr: str):
        self.workflow_name = workflow_name
        self.task_name = task_name
        self.err_metr = err_metr
        self.regressor = None
        self.train_X_scaler = None
        self.train_y_scaler = None
        self.X_train_full = None
        self.y_train_full = None
        self.model_error = None

    def initial_model_training(self, X_train, y_train) -> None:
        raise NotImplementedError("Model prediction method has not been implemented.")

    def predict_task(self, task_features: pd.Series) -> np.ndarray:
        raise NotImplementedError("Model prediction method has not been implemented.")

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> float:
        raise NotImplementedError("Predicting multiple tasks has not been implemented.")

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        raise NotImplementedError("Model update method has not been implemented.")
