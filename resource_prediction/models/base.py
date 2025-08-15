"""Base model interface for all prediction models."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict


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
        pass
    
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
    
    def set_params(self, **params) -> 'BasePredictor':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        return self