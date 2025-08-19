"""Additional Quantile Ensemble Predictor variants with different model combinations."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from typing import Dict, Any, Optional

from ..base import BasePredictor


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
        lgb_num_leaves: Optional[int] = None,
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
            lgb_params: Parameters for LightGBM regressor
            xgb_params: Parameters for XGBRegressor
            random_state: Random state for reproducibility
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build lgb_params
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": 1
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        # Override with individual parameters if provided
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
        # Build xgb_params
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
        """Fit both underlying regressors on the training data."""
        Xd = self._encode(X, fit=True)
        
        # Fit both models - LightGBM handles verbosity via its constructor params
        # Only pass fit_params that are actually supported by each model
        self.lgb.fit(Xd, y)
        
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        self.xgb.fit(Xd, y, **xgb_fit_params)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        
        lgb_preds = self.lgb.predict(Xd)
        xgb_preds = self.xgb.predict(Xd)
        preds = np.maximum(lgb_preds, xgb_preds)
        
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
        lgb_num_leaves: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        **kwargs
    ):
        """Initialize the GradientBoosting + LightGBM Quantile Ensemble Predictor."""
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build gb_params
        final_gb_params = {"loss": "quantile", "alpha": alpha, "random_state": random_state}
        if gb_params:
            final_gb_params.update(gb_params)
        
        if gb_n_estimators is not None:
            final_gb_params["n_estimators"] = gb_n_estimators
        if gb_max_depth is not None:
            final_gb_params["max_depth"] = gb_max_depth
        if gb_lr is not None:
            final_gb_params["learning_rate"] = gb_lr
            
        self.gb = GradientBoostingRegressor(**final_gb_params)
        
        # Build lgb_params
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": 1
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
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
        """Fit both underlying regressors on the training data."""
        Xd = self._encode(X, fit=True)
        
        self.gb.fit(Xd, y)
        
        # LightGBM handles verbosity via constructor params  
        self.lgb.fit(Xd, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        
        gb_preds = self.gb.predict(Xd)
        lgb_preds = self.lgb.predict(Xd)
        preds = np.maximum(gb_preds, lgb_preds)
        
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
        cat_iterations: Optional[int] = None,
        cat_depth: Optional[int] = None,
        cat_lr: Optional[float] = None,
        **kwargs
    ):
        """Initialize the XGBoost + CatBoost Quantile Ensemble Predictor."""
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build xgb_params
        final_xgb_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": alpha,
            "n_jobs": 1, 
            "random_state": random_state
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
        
        # Build cat_params
        final_cat_params = {
            "loss_function": f"Quantile:alpha={alpha}",
            "random_state": random_state,
            "verbose": False
        }
        if cat_params:
            final_cat_params.update(cat_params)
        
        if cat_iterations is not None:
            final_cat_params["iterations"] = cat_iterations
        if cat_depth is not None:
            final_cat_params["depth"] = cat_depth
        if cat_lr is not None:
            final_cat_params["learning_rate"] = cat_lr
            
        self.cat = cb.CatBoostRegressor(**final_cat_params)
        
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
        """Fit both underlying regressors on the training data."""
        Xd = self._encode(X, fit=True)
        
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        self.xgb.fit(Xd, y, **xgb_fit_params)
        
        # CatBoost handles verbosity via constructor params
        self.cat.fit(Xd, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        
        xgb_preds = self.xgb.predict(Xd)
        cat_preds = self.cat.predict(Xd)
        preds = np.maximum(xgb_preds, cat_preds)
        
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
        lgb_num_leaves: Optional[int] = None,
        lgb_lr: Optional[float] = None,
        cat_iterations: Optional[int] = None,
        cat_depth: Optional[int] = None,
        cat_lr: Optional[float] = None,
        **kwargs
    ):
        """Initialize the LightGBM + CatBoost Quantile Ensemble Predictor."""
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build lgb_params
        final_lgb_params = {
            "objective": "quantile", 
            "alpha": alpha,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": 1
        }
        if lgb_params:
            final_lgb_params.update(lgb_params)
        
        if lgb_n_estimators is not None:
            final_lgb_params["n_estimators"] = lgb_n_estimators
        if lgb_num_leaves is not None:
            final_lgb_params["num_leaves"] = lgb_num_leaves
        if lgb_lr is not None:
            final_lgb_params["learning_rate"] = lgb_lr
            
        self.lgb = lgb.LGBMRegressor(**final_lgb_params)
        
        # Build cat_params
        final_cat_params = {
            "loss_function": f"Quantile:alpha={alpha}",
            "random_state": random_state,
            "verbose": False
        }
        if cat_params:
            final_cat_params.update(cat_params)
        
        if cat_iterations is not None:
            final_cat_params["iterations"] = cat_iterations
        if cat_depth is not None:
            final_cat_params["depth"] = cat_depth
        if cat_lr is not None:
            final_cat_params["learning_rate"] = cat_lr
            
        self.cat = cb.CatBoostRegressor(**final_cat_params)
        
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
        """Fit both underlying regressors on the training data."""
        Xd = self._encode(X, fit=True)
        
        # LightGBM and CatBoost handle verbosity via constructor params
        self.lgb.fit(Xd, y)
        self.cat.fit(Xd, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by taking the maximum of both models and applying safety factor."""
        Xd = self._encode(X, fit=False)
        
        lgb_preds = self.lgb.predict(Xd)
        cat_preds = self.cat.predict(Xd)
        preds = np.maximum(lgb_preds, cat_preds)
        
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
    An ensemble regressor combining two specialized XGBoost quantile models.
    
    This model uses two XGBoost models with different configurations:
    - Conservative model: Higher quantile, deeper trees, fewer estimators (for complex patterns)
    - Aggressive model: Lower quantile, shallower trees, more estimators (for broad patterns)
    
    Takes the maximum prediction from both models and applies a safety factor.
    """
    
    def __init__(
        self, 
        alpha: float = 0.95, 
        safety: float = 1.05, 
        conservative_params: Optional[Dict[str, Any]] = None, 
        aggressive_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        # Individual parameters for conservative model
        conservative_quantile: Optional[float] = None,
        conservative_n_estimators: Optional[int] = None,
        conservative_max_depth: Optional[int] = None,
        conservative_lr: Optional[float] = None,
        # Individual parameters for aggressive model  
        aggressive_quantile: Optional[float] = None,
        aggressive_n_estimators: Optional[int] = None,
        aggressive_max_depth: Optional[int] = None,
        aggressive_lr: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize the XGBoost + XGBoost Specialized Quantile Ensemble Predictor.
        
        Args:
            alpha: Base quantile level (default: 0.95)
            safety: Safety multiplier applied to final predictions (default: 1.05)
            conservative_params: Parameters for conservative XGBoost model
            aggressive_params: Parameters for aggressive XGBoost model
            random_state: Random state for reproducibility
            **kwargs: Additional parameters (ignored gracefully)
        """
        self.alpha = alpha
        self.safety = safety
        self.random_state = random_state
        
        # Build conservative XGBoost params (higher quantile, deeper trees, fewer estimators)
        final_conservative_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": conservative_quantile if conservative_quantile is not None else min(0.99, alpha + 0.03),
            "n_jobs": 1, 
            "random_state": random_state
        }
        if conservative_params:
            final_conservative_params.update(conservative_params)
        
        # Override with individual parameters if provided
        if conservative_n_estimators is not None:
            final_conservative_params["n_estimators"] = conservative_n_estimators
        if conservative_max_depth is not None:
            final_conservative_params["max_depth"] = conservative_max_depth
        if conservative_lr is not None:
            final_conservative_params["learning_rate"] = conservative_lr
        if conservative_quantile is not None:
            final_conservative_params["quantile_alpha"] = conservative_quantile
            
        self.conservative_xgb = xgb.XGBRegressor(**final_conservative_params)
        
        # Build aggressive XGBoost params (lower quantile, shallower trees, more estimators)
        final_aggressive_params = {
            "objective": "reg:quantileerror", 
            "quantile_alpha": aggressive_quantile if aggressive_quantile is not None else max(0.85, alpha - 0.05),
            "n_jobs": 1, 
            "random_state": random_state
        }
        if aggressive_params:
            final_aggressive_params.update(aggressive_params)
        
        # Override with individual parameters if provided
        if aggressive_n_estimators is not None:
            final_aggressive_params["n_estimators"] = aggressive_n_estimators
        if aggressive_max_depth is not None:
            final_aggressive_params["max_depth"] = aggressive_max_depth
        if aggressive_lr is not None:
            final_aggressive_params["learning_rate"] = aggressive_lr
        if aggressive_quantile is not None:
            final_aggressive_params["quantile_alpha"] = aggressive_quantile
            
        self.aggressive_xgb = xgb.XGBRegressor(**final_aggressive_params)
        
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
        """Fit both specialized XGBoost regressors on the training data."""
        Xd = self._encode(X, fit=True)
        
        # Fit both models with quiet output
        xgb_fit_params = {"verbose": False}
        xgb_fit_params.update(fit_params)
        
        self.conservative_xgb.fit(Xd, y, **xgb_fit_params)
        self.aggressive_xgb.fit(Xd, y, **xgb_fit_params)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using specialized routing based on input characteristics.
        
        Conservative model is better for: complex builds, high memory, many dependencies
        Aggressive model is better for: standard builds, typical patterns
        """
        Xd = self._encode(X, fit=False)
        
        conservative_preds = self.conservative_xgb.predict(Xd)
        aggressive_preds = self.aggressive_xgb.predict(Xd)
        
        # Analyze input characteristics to determine model weights
        routing_weights = self._calculate_routing_weights(X)
        
        # Use weighted average instead of max, with safety bias towards conservative
        weighted_preds = (
            routing_weights * conservative_preds + 
            (1 - routing_weights) * aggressive_preds
        )
        
        # Apply safety bias: if conservative model predicts significantly higher, use it
        conservative_bias = conservative_preds > aggressive_preds * 1.2
        final_preds = np.where(
            conservative_bias,
            conservative_preds,  # Use conservative when it's significantly higher
            np.maximum(weighted_preds, aggressive_preds)  # Otherwise use weighted but at least aggressive
        )
        
        return final_preds * self.safety
    
    def _calculate_routing_weights(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate routing weights for each sample.
        Returns values between 0 and 1 where:
        - 1.0 = fully trust conservative model (complex/unusual builds)  
        - 0.0 = fully trust aggressive model (standard builds)
        """
        weights = np.zeros(len(X))
        
        # Feature-based complexity indicators
        complexity_score = 0.0
        
        # 1. Memory complexity (if present)
        if 'memory_limit' in X.columns:
            memory_values = X['memory_limit'].fillna(X['memory_limit'].median())
            # Normalize memory values and use high memory as complexity indicator
            memory_normalized = (memory_values - memory_values.min()) / (memory_values.max() - memory_values.min() + 1e-8)
            complexity_score += 0.3 * memory_normalized
            
        # 2. Build size/scale indicators  
        size_indicators = ['build_size', 'package_count', 'dependency_count', 'file_count']
        for col in size_indicators:
            if col in X.columns:
                values = X[col].fillna(X[col].median())
                # High values indicate complexity
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
                complexity_score += 0.1 * normalized
                
        # 3. Language/framework complexity
        complex_langs = ['cpp', 'c++', 'java', 'scala', 'rust']
        simple_langs = ['python', 'javascript', 'node']
        
        for col in X.columns:
            col_lower = col.lower()
            if any(lang in col_lower for lang in complex_langs):
                complexity_score += 0.1 * X[col].fillna(0)
            elif any(lang in col_lower for lang in simple_langs):
                complexity_score -= 0.05 * X[col].fillna(0)  # Subtract for simple languages
                
        # 4. Unusual feature combinations (high categorical diversity)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Count unique values across categorical features
            uniqueness_score = 0
            for col in categorical_cols:
                if len(X[col].unique()) > len(X) * 0.5:  # High uniqueness
                    uniqueness_score += 0.05
            complexity_score += uniqueness_score
            
        # 5. Extreme values detection (outliers indicate complexity)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = X[col].fillna(X[col].median())
            if len(values) > 1:
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    outliers = ((values < (q25 - 1.5 * iqr)) | (values > (q75 + 1.5 * iqr)))
                    complexity_score += 0.1 * outliers.astype(float)
        
        # Normalize and constrain to [0, 1]
        weights = np.clip(complexity_score, 0, 1)
        
        # Add some randomness for exploration but bias towards conservative for safety
        weights = weights * 0.8 + 0.2  # Minimum 20% conservative weight for safety
        
        return weights
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            conservative_params = self.conservative_xgb.get_params() if hasattr(self.conservative_xgb, 'get_params') else {}
        except Exception:
            conservative_params = {}
            
        try:
            aggressive_params = self.aggressive_xgb.get_params() if hasattr(self.aggressive_xgb, 'get_params') else {}
        except Exception:
            aggressive_params = {}
            
        return {
            "alpha": self.alpha,
            "safety": self.safety,
            "random_state": self.random_state,
            "conservative_params": conservative_params,
            "aggressive_params": aggressive_params
        }
    
    def set_params(self, **params) -> 'XGBXGBQuantileEnsemble':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self