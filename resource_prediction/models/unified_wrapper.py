"""Deployable Model Wrapper for production-ready model serialization with preprocessing."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from ..preprocessing import ModelPreprocessor
from .base import BasePredictor


class DeployableModel(BasePredictor):
    """
    A production-ready model wrapper that encapsulates trained models with preprocessing.

    This wrapper encapsulates:
    - The trained model
    - All preprocessing logic (feature engineering, encoding, etc.)
    - Model metadata (features, bin_edges, etc.)
    - Type-specific prediction logic

    Provides a single `predict(raw_dataframe)` interface for all models.
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        task_type: str,
        preprocessor: ModelPreprocessor,
        bin_edges: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the deployable model wrapper.

        Args:
            model: The trained model object
            model_type: Type of model ('lightgbm', 'xgboost', 'quantile_ensemble', etc.)
            task_type: Type of task ('classification', 'regression')
            preprocessor: Fitted preprocessing pipeline
            bin_edges: Bin edges for classification models
            metadata: Additional model metadata
        """
        self.model = model
        self.model_type = model_type
        self.task_type = task_type
        self.preprocessor = preprocessor
        self.bin_edges = bin_edges
        self.metadata = metadata or {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> "DeployableModel":
        """
        Fit the model and preprocessor to training data.

        Args:
            X: Training features
            y: Training targets
            **fit_params: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        # Fit preprocessor if not already fitted
        if not getattr(self.preprocessor, "is_fitted_", False):
            self.preprocessor.fit(X)

        # Preprocess the training data
        X_processed = self.preprocessor.transform(X)

        # Fit the underlying model
        self.model.fit(X_processed, y, **fit_params)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the memory allocation for the given input data.

        For classification models, if the model's confidence is below a
        tuned threshold, it predicts the next higher class.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            np.ndarray: The predicted memory allocations or classes.
        """
        # Preprocess the raw input data
        X_processed = self.preprocessor.transform(X)

        if self.task_type == "classification":
            # Classification model with one-up logic
            probabilities = self.model.predict_proba(X_processed)
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)

            # The confidence_threshold is mandatory and must be set on the model
            threshold = self.model.confidence_threshold

            low_confidence_indices = np.where(confidences < threshold)[0]

            if len(low_confidence_indices) > 0:
                max_class = len(self.model.classes_) - 1
                predictions[low_confidence_indices] = np.minimum(
                    predictions[low_confidence_indices] + 1, max_class
                )

            return predictions
        else:
            # Regression model
            return self.model.predict(X_processed)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the wrapped model.

        Returns:
            Dictionary containing model metadata
        """
        info = {
            "model_type": self.model_type,
            "task_type": self.task_type,
            **self.metadata,
        }

        # Add preprocessor info
        if self.preprocessor is not None and hasattr(
            self.preprocessor, "encoded_feature_names_"
        ):
            info["num_features"] = len(self.preprocessor.encoded_feature_names_)
            info["features"] = self.preprocessor.encoded_feature_names_.copy()

        # Add bin edges for classification models
        if self.bin_edges is not None:
            info["bin_edges"] = self.bin_edges.tolist()
            info["num_classes"] = len(self.bin_edges) - 1

        # Add model parameters if available
        if hasattr(self.model, "get_params"):
            try:
                info["model_params"] = self.model.get_params()
            except Exception:
                # Some models may have issues with get_params, skip it
                pass

        return info

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the deployable model wrapper to disk.

        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DeployableModel":
        """
        Load a deployable model wrapper from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded DeployableModel instance
        """
        return joblib.load(filepath)


def load_model(filepath: Union[str, Path]) -> DeployableModel:
    """
    Load a DeployableModel from disk.

    Args:
        filepath: Path to the deployable model file

    Returns:
        DeployableModel instance

    Raises:
        ValueError: If the file doesn't contain a DeployableModel
    """
    filepath = Path(filepath)

    try:
        model = joblib.load(filepath)
        if isinstance(model, DeployableModel):
            return model
        else:
            raise ValueError(f"File contains {type(model)}, expected DeployableModel")

    except Exception as e:
        raise ValueError(f"Failed to load deployable model from {filepath}: {e}")
