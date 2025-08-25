"""Quantile ensemble variants combining different model types."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from typing import Dict, Any, Optional

from ..base import BasePredictor


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
