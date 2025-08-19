"""Quantile Ensemble Predictor implementation."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from typing import Dict, Any, Optional

from ..base import BasePredictor


class QuantileEnsemblePredictor(BasePredictor):
    """
    An ensemble regressor combining GradientBoosting and XGBoost quantiles.
    
    This model takes the maximum prediction from both models and applies
    a safety factor to ensure conservative memory allocation.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        gb_params: Optional[Dict[str, Any]] = None, 
        xgb_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for convenience (will override gb_params/xgb_params if provided)
        gb_n_estimators: Optional[int] = None,
        gb_max_depth: Optional[int] = None,
        gb_lr: Optional[float] = None,  # Keep current naming
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,  # Keep current naming
        xgb_tree_method: str = 'hist',
        xgb_gpu_id: int = 0,
        **kwargs  # Accept any additional parameters gracefully
    ):
        """
        Initialize the Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            gb_params: Parameters for GradientBoostingRegressor
            xgb_params: Parameters for XGBRegressor
            random_state: Random state for reproducibility
            gb_n_estimators: Number of estimators for GradientBoosting
            gb_max_depth: Max depth for GradientBoosting
            gb_learning_rate: Learning rate for GradientBoosting
            xgb_n_estimators: Number of estimators for XGBoost
            xgb_max_depth: Max depth for XGBoost
            xgb_learning_rate: Learning rate for XGBoost
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build gb_params from individual parameters if provided
        final_gb_params = {"loss": "quantile", "alpha": alpha, "random_state": random_state}
        if gb_params:
            final_gb_params.update(gb_params)
        
        # Override with individual parameters if provided
        if gb_n_estimators is not None:
            final_gb_params["n_estimators"] = gb_n_estimators
        if gb_max_depth is not None:
            final_gb_params["max_depth"] = gb_max_depth
        if gb_lr is not None:
            final_gb_params["learning_rate"] = gb_lr
            
        self.gb = GradientBoostingRegressor(**final_gb_params)
        
        # Build xgb_params from individual parameters if provided
        final_xgb_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": alpha,
            "n_jobs": 1, 
            "random_state": random_state,
            "tree_method": xgb_tree_method
        }
        
        # Add GPU parameters for XGBoost if using GPU
        if xgb_tree_method == 'gpu_hist':
            final_xgb_params["gpu_id"] = xgb_gpu_id
        
        if xgb_params:
            final_xgb_params.update(xgb_params)
        
        # Override with individual parameters if provided
        if xgb_n_estimators is not None:
            final_xgb_params["n_estimators"] = xgb_n_estimators
        if xgb_max_depth is not None:
            final_xgb_params["max_depth"] = xgb_max_depth
        if xgb_lr is not None:
            final_xgb_params["learning_rate"] = xgb_lr
            
        self.xgb = xgb.XGBRegressor(**final_xgb_params)
        
        # Store column information for consistent encoding
        self.columns = None
    
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
        Fit both underlying regressors on the training data.
        
        Args:
            X: Feature matrix
            y: Target values
            **fit_params: Additional parameters passed to XGBoost fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # Fit both models
        self.gb.fit(Xd, y)
        
        # XGBoost fit parameters (verbose=False by default)
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        self.xgb.fit(Xd, y, **xgb_fit_params)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by taking the maximum of both models and applying safety factor.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions with safety factor applied
        """
        # Encode features using stored column information
        Xd = self._encode(X, fit=False)
        
        # Get predictions from both models and take maximum
        gb_preds = self.gb.predict(Xd)
        xgb_preds = self.xgb.predict(Xd)
        preds = np.maximum(gb_preds, xgb_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            gb_params = self.gb.get_params() if hasattr(self.gb, 'get_params') else {}
        except Exception:
            gb_params = {}
            
        try:
            xgb_params = self.xgb.get_params() if hasattr(self.xgb, 'get_params') else {}
        except Exception:
            xgb_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "gb_params": gb_params,
            "xgb_params": xgb_params
        }
    
    def set_params(self, **params) -> 'QuantileEnsemblePredictor':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self