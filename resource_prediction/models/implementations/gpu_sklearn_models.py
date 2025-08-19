"""GPU-accelerated alternatives for sklearn models using cuML and GPU-accelerated tree algorithms."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

from ..base import BasePredictor

# Try to import cuML for GPU-accelerated sklearn alternatives
try:
    from cuml.ensemble import RandomForestClassifier as CuRandomForestClassifier  
    from cuml.linear_model import LogisticRegression as CuLogisticRegression
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Alternative: Use XGBoost in RandomForest mode for GPU acceleration
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans


class GPURandomForestClassifier(BasePredictor):
    """
    GPU-accelerated Random Forest classifier using cuML if available,
    otherwise fallback to XGBoost in RandomForest mode with GPU acceleration.
    """
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'uniform',
        n_estimators: int = 300,
        max_depth: int = 10,
        random_state: int = 42,
        use_gpu: bool = True,
        gpu_id: int = 0,
        **kwargs
    ):
        """
        Initialize GPU-accelerated Random Forest classifier.
        
        Args:
            n_bins: Number of memory bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID
            **kwargs: Additional parameters
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        if CUML_AVAILABLE and use_gpu:
            # Use cuML GPU Random Forest
            rf_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': random_state,
            }
            rf_params.update(kwargs)
            self.model = CuRandomForestClassifier(**rf_params)
            self.backend = 'cuml'
        else:
            # Fallback to XGBoost in RandomForest mode with GPU support
            rf_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_state': random_state,
                'objective': 'multi:softprob',
                'tree_method': 'gpu_hist' if use_gpu else 'hist',
                'subsample': 0.8,  # RandomForest-like sampling
                'colsample_bytree': 0.8,  # RandomForest-like feature sampling
                'n_jobs': 1
            }
            if use_gpu:
                rf_params['gpu_id'] = gpu_id
            rf_params.update(kwargs)
            self.model = xgb.XGBClassifier(**rf_params)
            self.backend = 'xgboost_rf'
        
        self.bin_edges = None
        self.columns = None
    
    def _create_bins(self, y: pd.Series) -> np.ndarray:
        """Create memory bins based on strategy."""
        min_val, max_val = y.min(), y.max()
        
        if self.strategy == 'quantile':
            try:
                _, bin_edges = pd.qcut(y, q=self.n_bins, retbins=True, duplicates='drop')
            except ValueError:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        elif self.strategy == 'uniform':
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        else:  # kmeans
            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init='auto')
            kmeans.fit(y.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            edges = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
            bin_edges = np.array([min_val] + edges + [max_val])
        
        return np.array(sorted(list(set(bin_edges))))
    
    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encode categorical features."""
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False).astype(float)
        
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            
            # Align columns
            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0
            Xd = Xd[self.columns]
        
        return Xd
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit the GPU Random Forest classifier."""
        # Create bins
        self.bin_edges = self._create_bins(y)
        y_binned = pd.cut(y, bins=self.bin_edges, labels=False, include_lowest=True, right=True)
        
        # Encode features
        Xd = self._encode(X, fit=True)
        
        # Fit model
        if self.backend == 'cuml':
            # cuML requires numpy arrays
            self.model.fit(Xd.values, y_binned.values, **fit_params)
        else:
            # XGBoost accepts pandas
            fit_params.setdefault('verbose', False)
            self.model.fit(Xd, y_binned, **fit_params)
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> np.ndarray:
        """Make predictions with optional confidence threshold adjustment."""
        Xd = self._encode(X, fit=False)
        
        if self.backend == 'cuml':
            # cuML requires numpy arrays and returns different format
            pred_probs = self.model.predict_proba(Xd.values)
            pred_classes = np.argmax(pred_probs, axis=1)
        else:
            # XGBoost accepts pandas
            pred_probs = self.model.predict_proba(Xd)
            pred_classes = np.argmax(pred_probs, axis=1)
        
        # Apply confidence threshold adjustment
        adjusted_classes = []
        for i, probs in enumerate(pred_probs):
            pred_class = pred_classes[i]
            if probs[pred_class] < confidence_threshold:
                # Bump up to next class when uncertain
                pred_class = min(pred_class + 1, len(self.bin_edges) - 2)
            adjusted_classes.append(pred_class)
        
        # Convert class predictions to memory allocations
        return self.bin_edges[np.minimum(np.array(adjusted_classes) + 1, len(self.bin_edges) - 1)]
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {}
        try:
            params = self.model.get_params()
        except Exception:
            pass
        params.update({
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'backend': self.backend,
            'use_gpu': self.use_gpu
        })
        return params


class GPULogisticRegression(BasePredictor):
    """
    GPU-accelerated Logistic Regression using cuML if available,
    otherwise fallback to scikit-learn CPU implementation.
    """
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'uniform',
        C: float = 1.0,
        penalty: str = 'l2',
        random_state: int = 42,
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize GPU-accelerated Logistic Regression classifier.
        
        Args:
            n_bins: Number of memory bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            C: Regularization strength
            penalty: Penalty norm ('l1', 'l2', 'none')
            random_state: Random state for reproducibility
            use_gpu: Whether to use GPU acceleration
            **kwargs: Additional parameters
        """
        from sklearn.linear_model import LogisticRegression as SKLogisticRegression
        
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        if CUML_AVAILABLE and use_gpu:
            # Use cuML GPU Logistic Regression
            lr_params = {
                'C': C,
                'penalty': penalty,
                'random_state': random_state,
                'max_iter': 1000,
            }
            lr_params.update(kwargs)
            self.model = CuLogisticRegression(**lr_params)
            self.backend = 'cuml'
        else:
            # Fallback to scikit-learn CPU implementation
            lr_params = {
                'C': C,
                'penalty': penalty,
                'random_state': random_state,
                'max_iter': 1000,
                'solver': 'liblinear' if penalty in ['l1', 'l2'] else 'saga',
                'n_jobs': 1
            }
            lr_params.update(kwargs)
            self.model = SKLogisticRegression(**lr_params)
            self.backend = 'sklearn'
        
        self.bin_edges = None
        self.columns = None
    
    def _create_bins(self, y: pd.Series) -> np.ndarray:
        """Create memory bins based on strategy."""
        min_val, max_val = y.min(), y.max()
        
        if self.strategy == 'quantile':
            try:
                _, bin_edges = pd.qcut(y, q=self.n_bins, retbins=True, duplicates='drop')
            except ValueError:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        elif self.strategy == 'uniform':
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        else:  # kmeans
            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init='auto')
            kmeans.fit(y.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            edges = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
            bin_edges = np.array([min_val] + edges + [max_val])
        
        return np.array(sorted(list(set(bin_edges))))
    
    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """One-hot encode categorical features."""
        Xd = pd.get_dummies(X, drop_first=True, dummy_na=False).astype(float)
        
        if fit:
            self.columns = Xd.columns.tolist()
        else:
            if self.columns is None:
                raise ValueError("Model must be fitted before making predictions")
            
            # Align columns
            missing_cols = set(self.columns) - set(Xd.columns)
            for col in missing_cols:
                Xd[col] = 0
            Xd = Xd[self.columns]
        
        return Xd
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit the GPU Logistic Regression classifier."""
        # Create bins
        self.bin_edges = self._create_bins(y)
        y_binned = pd.cut(y, bins=self.bin_edges, labels=False, include_lowest=True, right=True)
        
        # Encode features
        Xd = self._encode(X, fit=True)
        
        # Fit model
        if self.backend == 'cuml':
            # cuML requires numpy arrays
            self.model.fit(Xd.values, y_binned.values, **fit_params)
        else:
            # Scikit-learn accepts pandas
            self.model.fit(Xd, y_binned, **fit_params)
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> np.ndarray:
        """Make predictions with optional confidence threshold adjustment."""
        Xd = self._encode(X, fit=False)
        
        if self.backend == 'cuml':
            # cuML requires numpy arrays
            pred_probs = self.model.predict_proba(Xd.values)
            pred_classes = np.argmax(pred_probs, axis=1)
        else:
            # Scikit-learn accepts pandas
            pred_probs = self.model.predict_proba(Xd)
            pred_classes = np.argmax(pred_probs, axis=1)
        
        # Apply confidence threshold adjustment
        adjusted_classes = []
        for i, probs in enumerate(pred_probs):
            pred_class = pred_classes[i]
            if probs[pred_class] < confidence_threshold:
                # Bump up to next class when uncertain
                pred_class = min(pred_class + 1, len(self.bin_edges) - 2)
            adjusted_classes.append(pred_class)
        
        # Convert class predictions to memory allocations
        return self.bin_edges[np.minimum(np.array(adjusted_classes) + 1, len(self.bin_edges) - 1)]
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {}
        try:
            params = self.model.get_params()
        except Exception:
            pass
        params.update({
            'n_bins': self.n_bins,
            'strategy': self.strategy,
            'backend': self.backend,
            'use_gpu': self.use_gpu
        })
        return params