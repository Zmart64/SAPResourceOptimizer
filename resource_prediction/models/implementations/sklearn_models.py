"""Sklearn model wrappers implementing BasePredictor interface."""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier as SKRandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.cluster import KMeans
from typing import Dict, Any

from ..base import BasePredictor


class RandomForestClassifier(BasePredictor):
    """Random Forest Classifier wrapper for memory bin classification."""
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'uniform',
        n_estimators: int = 300,
        max_depth: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_bins: Number of memory bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
            **kwargs: Additional Random Forest parameters
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'n_jobs': 1
        }
        rf_params.update(kwargs)
        
        self.model = SKRandomForestClassifier(**rf_params)
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
    
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit the Random Forest classifier."""
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


class LogisticRegression(BasePredictor):
    """Logistic Regression wrapper for memory bin classification."""
    
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'uniform',
        C: float = 1.0,
        solver: str = 'liblinear',
        penalty: str = 'l2',
        l1_ratio: float = 0.5,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Logistic Regression classifier.
        
        Args:
            n_bins: Number of memory bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            C: Regularization strength
            solver: Algorithm to use in the optimization problem
            penalty: Penalty norm ('l1', 'l2', 'elasticnet', 'none')
            l1_ratio: Elastic-Net mixing parameter (only used if penalty='elasticnet')
            random_state: Random state for reproducibility
            **kwargs: Additional Logistic Regression parameters
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        
        lr_params = {
            'C': C,
            'solver': solver,
            'penalty': penalty,
            'random_state': random_state,
            'max_iter': 1000,
            'n_jobs': 1
        }
        
        # Only add l1_ratio if penalty is elasticnet
        if penalty == 'elasticnet':
            lr_params['l1_ratio'] = l1_ratio
        
        lr_params.update(kwargs)
        
        self.model = SKLogisticRegression(**lr_params)
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
    
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit the Logistic Regression classifier."""
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
