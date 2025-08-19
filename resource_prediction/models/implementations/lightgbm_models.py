"""LightGBM model wrappers implementing BasePredictor interface."""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from typing import Dict, Any

from ..base import BasePredictor


class LightGBMRegressor(BasePredictor):
    """LightGBM Regressor wrapper for quantile regression."""
    
    def __init__(
        self, 
        alpha: float = 0.95,
        n_estimators: int = 400,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        random_state: int = 42,
        device_type: str = 'cpu',
        gpu_device_id: int = 0,
        **kwargs
    ):
        """
        Initialize LightGBM quantile regressor.
        
        Args:
            alpha: Quantile level for regression
            n_estimators: Number of boosting rounds
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate
            random_state: Random state for reproducibility
            device_type: Device type ('cpu' or 'gpu')
            gpu_device_id: GPU device ID (0 by default)
            **kwargs: Additional LightGBM parameters
        """
        self.alpha = alpha
        self.random_state = random_state
        
        # Set up LightGBM parameters for quantile regression
        lgb_params = {
            'objective': 'quantile',
            'alpha': alpha,
            'n_estimators': n_estimators,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,
            'device_type': device_type,
            'n_jobs': 1
        }
        
        # Add GPU parameters if using GPU device
        if device_type == 'gpu':
            lgb_params['gpu_device_id'] = gpu_device_id
        
        lgb_params.update(kwargs)
        
        self.model = lgb.LGBMRegressor(**lgb_params)
        self.columns = None
    
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
        """Fit the LightGBM model."""
        Xd = self._encode(X, fit=True)
        self.model.fit(Xd, y, **fit_params)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        Xd = self._encode(X, fit=False)
        return self.model.predict(Xd)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()


class LightGBMClassifier(BasePredictor):
    """LightGBM Classifier wrapper for memory bin classification."""
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'uniform',
        n_estimators: int = 400,
        max_depth: int = 6,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        random_state: int = 42,
        device_type: str = 'cpu',
        gpu_device_id: int = 0,
        **kwargs
    ):
        """
        Initialize LightGBM classifier.
        
        Args:
            n_bins: Number of memory bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate
            random_state: Random state for reproducibility
            device_type: Device type ('cpu' or 'gpu')
            gpu_device_id: GPU device ID (0 by default)
            **kwargs: Additional LightGBM parameters
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        
        lgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': -1,
            'device_type': device_type,
            'n_jobs': 1
        }
        
        # Add GPU parameters if using GPU device
        if device_type == 'gpu':
            lgb_params['gpu_device_id'] = gpu_device_id
        
        lgb_params.update(kwargs)
        
        self.model = lgb.LGBMClassifier(**lgb_params)
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
        """Fit the LightGBM classifier."""
        # Create bins
        self.bin_edges = self._create_bins(y)
        y_binned = pd.cut(y, bins=self.bin_edges, labels=False, include_lowest=True, right=True)
        
        # Encode features
        Xd = self._encode(X, fit=True)
        
        # Fit model
        self.model.fit(Xd, y_binned, **fit_params)
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> np.ndarray:
        """
        Make predictions with optional confidence threshold adjustment.
        
        Args:
            X: Input features
            confidence_threshold: When model confidence is below this threshold, 
                                 bump up to next higher memory class for safety
                                 
        Returns:
            Memory allocation predictions in GB
        """
        Xd = self._encode(X, fit=False)
        
        # Get probability predictions for confidence assessment
        pred_probs = self.model.predict_proba(Xd)
        pred_classes = np.argmax(pred_probs, axis=1)
        
        # Apply confidence threshold adjustment - key business logic!
        # If model is uncertain, allocate more memory for safety
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
        params = self.model.get_params()
        params.update({
            'n_bins': self.n_bins,
            'strategy': self.strategy
        })
        return params
