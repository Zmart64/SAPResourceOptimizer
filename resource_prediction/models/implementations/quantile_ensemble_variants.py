"""Quantile ensemble variants combining different model types."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from typing import Dict, Any, Optional

from ..base import BasePredictor
from .sklearn_models import RandomForestQuantileRegressor


class GBXGBQuantileEnsemble(BasePredictor):
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
        **kwargs  # Accept any additional parameters gracefully
    ):
        """
        Initialize the GradientBoosting + XGBoost Quantile Ensemble Predictor.
        
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
            "random_state": random_state
        }
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
    
    def set_params(self, **params) -> 'GBXGBQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class LGBXGBQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining LightGBM and XGBoost quantiles.
    
    This model takes the maximum prediction from both models and applies
    a safety factor to ensure conservative memory allocation.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        lgb_params: Optional[Dict[str, Any]] = None, 
        xgb_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for convenience
        lgb_n_estimators: Optional[int] = None,
        lgb_max_depth: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        lgb_num_leaves: Optional[int] = None,
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the LightGBM + XGBoost Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            lgb_params: Parameters for LGBMRegressor
            xgb_params: Parameters for XGBRegressor
            random_state: Random state for reproducibility
            lgb_n_estimators: Number of estimators for LightGBM
            lgb_max_depth: Max depth for LightGBM
            lgb_lr: Learning rate for LightGBM
            xgb_n_estimators: Number of estimators for XGBoost
            xgb_max_depth: Max depth for XGBoost
            xgb_lr: Learning rate for XGBoost
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build lgb_params from individual parameters if provided
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha, 
            "verbose": -1, 
            "random_state": random_state
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        # Override with individual parameters if provided
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_max_depth is not None:
            final_lgb_params["max_depth"] = lgb_max_depth
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
        # Build xgb_params from individual parameters if provided
        final_xgb_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": alpha,
            "n_jobs": 1, 
            "random_state": random_state
        }
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
            **fit_params: Additional parameters passed to model fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # Fit both models
        self.lgb.fit(Xd, y)
        
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
        lgb_preds = self.lgb.predict(Xd)
        xgb_preds = self.xgb.predict(Xd)
        preds = np.maximum(lgb_preds, xgb_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            lgb_params = self.lgb.get_params() if hasattr(self.lgb, 'get_params') else {}
        except Exception:
            lgb_params = {}
            
        try:
            xgb_params = self.xgb.get_params() if hasattr(self.xgb, 'get_params') else {}
        except Exception:
            xgb_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "lgb_params": lgb_params,
            "xgb_params": xgb_params
        }
    
    def set_params(self, **params) -> 'LGBXGBQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class GBLGBQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining GradientBoosting and LightGBM quantiles.
    
    This model takes the maximum prediction from both models and applies
    a safety factor to ensure conservative memory allocation.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        gb_params: Optional[Dict[str, Any]] = None, 
        lgb_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for convenience
        gb_n_estimators: Optional[int] = None,
        gb_max_depth: Optional[int] = None,
        gb_lr: Optional[float] = None,
        lgb_n_estimators: Optional[int] = None,
        lgb_max_depth: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        lgb_num_leaves: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the GradientBoosting + LightGBM Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            gb_params: Parameters for GradientBoostingRegressor
            lgb_params: Parameters for LGBMRegressor
            random_state: Random state for reproducibility
            gb_n_estimators: Number of estimators for GradientBoosting
            gb_max_depth: Max depth for GradientBoosting
            gb_lr: Learning rate for GradientBoosting
            lgb_n_estimators: Number of estimators for LightGBM
            lgb_max_depth: Max depth for LightGBM
            lgb_lr: Learning rate for LightGBM
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
        
        # Build lgb_params from individual parameters if provided
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha, 
            "verbose": -1, 
            "random_state": random_state
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        # Override with individual parameters if provided
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_max_depth is not None:
            final_lgb_params["max_depth"] = lgb_max_depth
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
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
            **fit_params: Additional parameters passed to model fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # Fit both models
        self.gb.fit(Xd, y)
        self.lgb.fit(Xd, y)
    
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
        lgb_preds = self.lgb.predict(Xd)
        preds = np.maximum(gb_preds, lgb_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            gb_params = self.gb.get_params() if hasattr(self.gb, 'get_params') else {}
        except Exception:
            gb_params = {}
            
        try:
            lgb_params = self.lgb.get_params() if hasattr(self.lgb, 'get_params') else {}
        except Exception:
            lgb_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "gb_params": gb_params,
            "lgb_params": lgb_params
        }
    
    def set_params(self, **params) -> 'GBLGBQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class XGBCatQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining XGBoost and CatBoost quantiles.
    
    This model takes the maximum prediction from both models and applies
    a safety factor to ensure conservative memory allocation.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        xgb_params: Optional[Dict[str, Any]] = None, 
        cat_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for convenience
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,
        cat_n_estimators: Optional[int] = None,
        cat_max_depth: Optional[int] = None,
        cat_lr: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the XGBoost + CatBoost Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            xgb_params: Parameters for XGBRegressor
            cat_params: Parameters for CatBoostRegressor
            random_state: Random state for reproducibility
            xgb_n_estimators: Number of estimators for XGBoost
            xgb_max_depth: Max depth for XGBoost
            xgb_lr: Learning rate for XGBoost
            cat_n_estimators: Number of estimators for CatBoost
            cat_max_depth: Max depth for CatBoost
            cat_lr: Learning rate for CatBoost
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build xgb_params from individual parameters if provided
        final_xgb_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": alpha,
            "n_jobs": 1, 
            "random_state": random_state
        }
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
        
        # Build cat_params from individual parameters if provided
        final_cat_params = {
            "loss_function": f"Quantile:alpha={alpha}",
            "verbose": False, 
            "random_state": random_state
        }
        if cat_params:
            final_cat_params.update(cat_params)
        
        # Override with individual parameters if provided
        if cat_n_estimators is not None:
            final_cat_params["n_estimators"] = cat_n_estimators
        if cat_max_depth is not None:
            final_cat_params["max_depth"] = cat_max_depth
        if cat_lr is not None:
            final_cat_params["learning_rate"] = cat_lr
            
        self.cat = cat.CatBoostRegressor(**final_cat_params)
        
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
            **fit_params: Additional parameters passed to model fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # Fit both models
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        self.xgb.fit(Xd, y, **xgb_fit_params)
        self.cat.fit(Xd, y)
    
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
        xgb_preds = self.xgb.predict(Xd)
        cat_preds = self.cat.predict(Xd)
        preds = np.maximum(xgb_preds, cat_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            xgb_params = self.xgb.get_params() if hasattr(self.xgb, 'get_params') else {}
        except Exception:
            xgb_params = {}
            
        try:
            cat_params = self.cat.get_params() if hasattr(self.cat, 'get_params') else {}
        except Exception:
            cat_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "xgb_params": xgb_params,
            "cat_params": cat_params
        }
    
    def set_params(self, **params) -> 'XGBCatQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class LGBCatQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining LightGBM and CatBoost quantiles.
    
    This model takes the maximum prediction from both models and applies
    a safety factor to ensure conservative memory allocation.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        lgb_params: Optional[Dict[str, Any]] = None, 
        cat_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for convenience
        lgb_n_estimators: Optional[int] = None,
        lgb_max_depth: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        lgb_num_leaves: Optional[int] = None,
        cat_n_estimators: Optional[int] = None,
        cat_max_depth: Optional[int] = None,
        cat_lr: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the LightGBM + CatBoost Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            lgb_params: Parameters for LGBMRegressor
            cat_params: Parameters for CatBoostRegressor
            random_state: Random state for reproducibility
            lgb_n_estimators: Number of estimators for LightGBM
            lgb_max_depth: Max depth for LightGBM
            lgb_lr: Learning rate for LightGBM
            cat_n_estimators: Number of estimators for CatBoost
            cat_max_depth: Max depth for CatBoost
            cat_lr: Learning rate for CatBoost
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build lgb_params from individual parameters if provided
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha, 
            "verbose": -1, 
            "random_state": random_state
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        # Override with individual parameters if provided
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_max_depth is not None:
            final_lgb_params["max_depth"] = lgb_max_depth
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
        # Build cat_params from individual parameters if provided
        final_cat_params = {
            "loss_function": f"Quantile:alpha={alpha}",
            "verbose": False, 
            "random_state": random_state
        }
        if cat_params:
            final_cat_params.update(cat_params)
        
        # Override with individual parameters if provided
        if cat_n_estimators is not None:
            final_cat_params["n_estimators"] = cat_n_estimators
        if cat_max_depth is not None:
            final_cat_params["max_depth"] = cat_max_depth
        if cat_lr is not None:
            final_cat_params["learning_rate"] = cat_lr
            
        self.cat = cat.CatBoostRegressor(**final_cat_params)
        
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
            **fit_params: Additional parameters passed to model fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # Fit both models
        self.lgb.fit(Xd, y)
        self.cat.fit(Xd, y)
    
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
        lgb_preds = self.lgb.predict(Xd)
        cat_preds = self.cat.predict(Xd)
        preds = np.maximum(lgb_preds, cat_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            lgb_params = self.lgb.get_params() if hasattr(self.lgb, 'get_params') else {}
        except Exception:
            lgb_params = {}
            
        try:
            cat_params = self.cat.get_params() if hasattr(self.cat, 'get_params') else {}
        except Exception:
            cat_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "lgb_params": lgb_params,
            "cat_params": cat_params
        }
    
    def set_params(self, **params) -> 'LGBCatQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class XGBXGBQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining two XGBoost models with standard parameter ranges.
    
    Uses two XGBoost models with the same quantile level but different hyperparameters.
    Takes the maximum prediction from both models and applies a safety factor.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        random_state: int = 42,
        # Standard model parameters (both models use same quantile)
        xgb1_n_estimators: int = 300,
        xgb1_max_depth: int = 6,
        xgb1_lr: float = 0.05,
        xgb2_n_estimators: int = 500,
        xgb2_max_depth: int = 8,
        xgb2_lr: float = 0.03,
        **kwargs
    ):
        """
        Initialize the XGBoost + XGBoost Standard Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            random_state: Random state for reproducibility
            xgb1_n_estimators: Number of estimators for first XGBoost model
            xgb1_max_depth: Max depth for first XGBoost model
            xgb1_lr: Learning rate for first XGBoost model
            xgb2_n_estimators: Number of estimators for second XGBoost model
            xgb2_max_depth: Max depth for second XGBoost model
            xgb2_lr: Learning rate for second XGBoost model
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # First XGBoost model
        self.xgb1 = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=xgb1_n_estimators,
            max_depth=xgb1_max_depth,
            learning_rate=xgb1_lr,
            n_jobs=1,
            random_state=random_state
        )
        
        # Second XGBoost model
        self.xgb2 = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=xgb2_n_estimators,
            max_depth=xgb2_max_depth,
            learning_rate=xgb2_lr,
            n_jobs=1,
            random_state=random_state + 1  # Slightly different random state for diversity
        )
        
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
        Fit both underlying XGBoost regressors on the training data.
        
        Args:
            X: Feature matrix
            y: Target values
            **fit_params: Additional parameters passed to XGBoost fit
        """
        # Encode features and store column information
        Xd = self._encode(X, fit=True)
        
        # XGBoost fit parameters (verbose=False by default)
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        
        # Fit both models
        self.xgb1.fit(Xd, y, **xgb_fit_params)
        self.xgb2.fit(Xd, y, **xgb_fit_params)
    
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
        xgb1_preds = self.xgb1.predict(Xd)
        xgb2_preds = self.xgb2.predict(Xd)
        preds = np.maximum(xgb1_preds, xgb2_preds)
        
        # Apply safety factor
        return preds * self.safety
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            xgb1_params = self.xgb1.get_params() if hasattr(self.xgb1, 'get_params') else {}
        except Exception:
            xgb1_params = {}
            
        try:
            xgb2_params = self.xgb2.get_params() if hasattr(self.xgb2, 'get_params') else {}
        except Exception:
            xgb2_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "xgb1_params": xgb1_params,
            "xgb2_params": xgb2_params
        }
    
    def set_params(self, **params) -> 'XGBXGBQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class LGBRFQuantileEnsemble(BasePredictor):
    """
    Ensemble regressor combining LightGBM quantile and RandomForest quantile.

    Takes the maximum prediction across models and applies a safety factor.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        safety: float = 1.05,
        lgb_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # convenience knobs
        lgb_n_estimators: Optional[int] = None,
        lgb_max_depth: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        lgb_num_leaves: Optional[int] = None,
        rf_n_estimators: Optional[int] = None,
        rf_max_depth: Optional[int] = None,
        rf_min_samples_leaf: Optional[int] = None,
        **kwargs,
    ):
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state

        final_lgb_params = {
            "objective": "quantile",
            "alpha": alpha,
            "verbose": -1,
            "random_state": random_state,
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_max_depth is not None:
            final_lgb_params["max_depth"] = lgb_max_depth
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)

        final_rf_params = {
            "alpha": alpha,
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "random_state": random_state + 1,
        }
        if rf_params:
            final_rf_params.update(rf_params)
        if rf_n_estimators is not None:
            final_rf_params["n_estimators"] = rf_n_estimators
        if rf_max_depth is not None:
            final_rf_params["max_depth"] = rf_max_depth
        if rf_min_samples_leaf is not None:
            final_rf_params["min_samples_leaf"] = rf_min_samples_leaf
        self.rf = RandomForestQuantileRegressor(**final_rf_params)

        self.columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            missing = set(self.columns) - set(Xd.columns)
            for c in missing:
                Xd[c] = 0
            Xd = Xd[self.columns]
        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        Xd = self._encode(X, fit=True)
        self.lgb.fit(Xd, y)
        self.rf.fit(X, y)  # rf encodes internally again; safe and consistent

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xd = self._encode(X, fit=False)
        p_lgb = self.lgb.predict(Xd)
        p_rf = self.rf.predict(X)
        return np.maximum(p_lgb, p_rf) * self.safety

    def get_params(self) -> Dict[str, Any]:
        try:
            lp = self.lgb.get_params()
        except Exception:
            lp = {}
        try:
            rp = self.rf.get_params()
        except Exception:
            rp = {}
        return {"alpha": self.alpha, "safety": self.safety, "random_state": self.random_state, "lgb_params": lp, "rf_params": rp}

    def set_params(self, **params) -> "LGBRFQuantileEnsemble":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


class XGBRFQuantileEnsemble(BasePredictor):
    """
    Ensemble regressor combining XGBoost quantile and RandomForest quantile.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        safety: float = 1.05,
        xgb_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # xgb
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,
        # rf
        rf_n_estimators: Optional[int] = None,
        rf_max_depth: Optional[int] = None,
        rf_min_samples_leaf: Optional[int] = None,
        **kwargs,
    ):
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state

        final_xgb_params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": alpha,
            "n_jobs": 1,
            "random_state": random_state,
        }
        if xgb_params:
            final_xgb_params.update(xgb_params)
        if xgb_n_estimators is not None:
            final_xgb_params["n_estimators"] = xgb_n_estimators
        if xgb_max_depth is not None:
            final_xgb_params["max_depth"] = xgb_max_depth
        if xgb_lr is not None:
            final_xgb_params["learning_rate"] = xgb_lr
        self.xgb = xgb.XGBRegressor(**final_xgb_params)

        final_rf_params = {
            "alpha": alpha,
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "random_state": random_state + 1,
        }
        if rf_params:
            final_rf_params.update(rf_params)
        if rf_n_estimators is not None:
            final_rf_params["n_estimators"] = rf_n_estimators
        if rf_max_depth is not None:
            final_rf_params["max_depth"] = rf_max_depth
        if rf_min_samples_leaf is not None:
            final_rf_params["min_samples_leaf"] = rf_min_samples_leaf
        self.rf = RandomForestQuantileRegressor(**final_rf_params)

        self.columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            missing = set(self.columns) - set(Xd.columns)
            for c in missing:
                Xd[c] = 0
            Xd = Xd[self.columns]
        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        Xd = self._encode(X, fit=True)
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        self.xgb.fit(Xd, y, **xgb_fit_params)
        self.rf.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xd = self._encode(X, fit=False)
        p_xgb = self.xgb.predict(Xd)
        p_rf = self.rf.predict(X)
        return np.maximum(p_xgb, p_rf) * self.safety

    def get_params(self) -> Dict[str, Any]:
        try:
            xp = self.xgb.get_params()
        except Exception:
            xp = {}
        try:
            rp = self.rf.get_params()
        except Exception:
            rp = {}
        return {"alpha": self.alpha, "safety": self.safety, "random_state": self.random_state, "xgb_params": xp, "rf_params": rp}

    def set_params(self, **params) -> "XGBRFQuantileEnsemble":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


class RFRFQuantileEnsemble(BasePredictor):
    """
    Ensemble regressor combining two RandomForest quantile models.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        safety: float = 1.05,
        random_state: int = 42,
        # RF1
        rf1_n_estimators: int = 300,
        rf1_max_depth: Optional[int] = 6,
        rf1_min_samples_leaf: int = 1,
        # RF2
        rf2_n_estimators: int = 500,
        rf2_max_depth: Optional[int] = 8,
        rf2_min_samples_leaf: int = 1,
        **kwargs,
    ):
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state

        self.rf1 = RandomForestQuantileRegressor(
            alpha=alpha,
            n_estimators=rf1_n_estimators,
            max_depth=rf1_max_depth,
            min_samples_leaf=rf1_min_samples_leaf,
            random_state=random_state,
        )
        self.rf2 = RandomForestQuantileRegressor(
            alpha=alpha,
            n_estimators=rf2_n_estimators,
            max_depth=rf2_max_depth,
            min_samples_leaf=rf2_min_samples_leaf,
            random_state=random_state + 1,
        )

        self.columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            missing = set(self.columns) - set(Xd.columns)
            for c in missing:
                Xd[c] = 0
            Xd = Xd[self.columns]
        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        # RF models encode internally; use original X
        self.rf1.fit(X, y)
        self.rf2.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.rf1.predict(X)
        p2 = self.rf2.predict(X)
        return np.maximum(p1, p2) * self.safety

    def get_params(self) -> Dict[str, Any]:
        try:
            p1 = self.rf1.get_params()
        except Exception:
            p1 = {}
        try:
            p2 = self.rf2.get_params()
        except Exception:
            p2 = {}
        return {"alpha": self.alpha, "safety": self.safety, "random_state": self.random_state, "rf1_params": p1, "rf2_params": p2}

    def set_params(self, **params) -> "RFRFQuantileEnsemble":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


class LGBLGBQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining two LightGBM models with standard parameter ranges.
    
    Uses two LightGBM models with the same quantile level but different hyperparameters.
    Takes the maximum prediction from both models and applies a safety factor.
    """
    
    def __init__(
        self,
        alpha: float = 0.95,
        safety: float = 1.05,
        random_state: int = 42,
        # First LGBM model params
        lgb1_n_estimators: int = 300,
        lgb1_max_depth: Optional[int] = 6,
        lgb1_lr: float = 0.05,
        lgb1_num_leaves: Optional[int] = None,
        # Second LGBM model params
        lgb2_n_estimators: int = 500,
        lgb2_max_depth: Optional[int] = 8,
        lgb2_lr: float = 0.03,
        lgb2_num_leaves: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the LightGBM + LightGBM Standard Quantile Ensemble Predictor.
        
        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            random_state: Random state for reproducibility
            lgb1_n_estimators: Number of estimators for first LightGBM model
            lgb1_max_depth: Max depth for first LightGBM model (None to use LightGBM default)
            lgb1_lr: Learning rate for first LightGBM model
            lgb1_num_leaves: Num leaves for first LightGBM model (optional)
            lgb2_n_estimators: Number of estimators for second LightGBM model
            lgb2_max_depth: Max depth for second LightGBM model (None to use LightGBM default)
            lgb2_lr: Learning rate for second LightGBM model
            lgb2_num_leaves: Num leaves for second LightGBM model (optional)
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state

        # First LightGBM model
        lgb1_params = {
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": lgb1_n_estimators,
            "learning_rate": lgb1_lr,
            "verbose": -1,
            "random_state": random_state,
        }
        if lgb1_max_depth is not None:
            lgb1_params["max_depth"] = lgb1_max_depth
        if lgb1_num_leaves is not None:
            lgb1_params["num_leaves"] = lgb1_num_leaves
        self.lgb1 = lgb.LGBMRegressor(**lgb1_params)

        # Second LightGBM model
        lgb2_params = {
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": lgb2_n_estimators,
            "learning_rate": lgb2_lr,
            "verbose": -1,
            "random_state": random_state + 1,
        }
        if lgb2_max_depth is not None:
            lgb2_params["max_depth"] = lgb2_max_depth
        if lgb2_num_leaves is not None:
            lgb2_params["num_leaves"] = lgb2_num_leaves
        self.lgb2 = lgb.LGBMRegressor(**lgb2_params)

        # Store column information for consistent encoding
        self.columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical features and align columns.
        """
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0
            Xd = Xd[self.columns]
        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit both underlying LightGBM regressors on the training data."""
        Xd = self._encode(X, fit=True)
        self.lgb1.fit(Xd, y)
        self.lgb2.fit(Xd, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        p1 = self.lgb1.predict(Xd)
        p2 = self.lgb2.predict(Xd)
        return np.maximum(p1, p2) * self.safety

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            lgb1_params = self.lgb1.get_params() if hasattr(self.lgb1, "get_params") else {}
        except Exception:
            lgb1_params = {}
        try:
            lgb2_params = self.lgb2.get_params() if hasattr(self.lgb2, "get_params") else {}
        except Exception:
            lgb2_params = {}
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "lgb1_params": lgb1_params,
            "lgb2_params": lgb2_params,
        }

    def set_params(self, **params) -> "LGBLGBQuantileEnsemble":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


class CatCatQuantileEnsemble(BasePredictor):
    """
    An ensemble regressor combining two CatBoost models with standard parameter ranges.
    
    Uses two CatBoost models with the same quantile level but different hyperparameters.
    Takes the maximum prediction from both models and applies a safety factor.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        safety: float = 1.05,
        random_state: int = 42,
        # First CatBoost model params
        cat1_iterations: int = 300,
        cat1_depth: int = 6,
        cat1_lr: float = 0.05,
        # Second CatBoost model params
        cat2_iterations: int = 500,
        cat2_depth: int = 8,
        cat2_lr: float = 0.03,
        **kwargs,
    ):
        """
        Initialize the CatBoost + CatBoost Standard Quantile Ensemble Predictor.

        Args:
            alpha: Quantile level for both models (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            random_state: Random state for reproducibility
            cat1_iterations: Iterations for first CatBoost model
            cat1_depth: Max depth for first CatBoost model
            cat1_lr: Learning rate for first CatBoost model
            cat2_iterations: Iterations for second CatBoost model
            cat2_depth: Max depth for second CatBoost model
            cat2_lr: Learning rate for second CatBoost model
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state

        # First CatBoost model
        self.cat1 = cat.CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            iterations=cat1_iterations,
            depth=cat1_depth,
            learning_rate=cat1_lr,
            verbose=False,
            random_state=random_state,
        )

        # Second CatBoost model
        self.cat2 = cat.CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}",
            iterations=cat2_iterations,
            depth=cat2_depth,
            learning_rate=cat2_lr,
            verbose=False,
            random_state=random_state + 1,
        )

        # Store column information for consistent encoding
        self.columns = None

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encode categorical features and align columns."""
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0
            Xd = Xd[self.columns]
        return Xd.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit both underlying CatBoost regressors on the training data."""
        Xd = self._encode(X, fit=True)
        self.cat1.fit(Xd, y)
        self.cat2.fit(Xd, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        p1 = self.cat1.predict(Xd)
        p2 = self.cat2.predict(Xd)
        return np.maximum(p1, p2) * self.safety

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            cat1_params = self.cat1.get_params() if hasattr(self.cat1, "get_params") else {}
        except Exception:
            cat1_params = {}
        try:
            cat2_params = self.cat2.get_params() if hasattr(self.cat2, "get_params") else {}
        except Exception:
            cat2_params = {}
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "cat1_params": cat1_params,
            "cat2_params": cat2_params,
        }

    def set_params(self, **params) -> "CatCatQuantileEnsemble":
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
