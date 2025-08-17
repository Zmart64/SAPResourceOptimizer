"""Unified Model Wrapper for consistent interface across all model types."""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import os

from .quantile_ensemble import QuantileEnsemblePredictor


class UnifiedModelWrapper:
    """
    A unified wrapper that provides a consistent interface for all model types.
    
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
        features: List[str],
        bin_edges: Optional[np.ndarray] = None,
        preprocessing_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the unified model wrapper.
        
        Args:
            model: The trained model object
            model_type: Type of model ('lightgbm', 'xgboost', 'quantile_ensemble', etc.)
            task_type: Type of task ('classification', 'regression')
            features: List of feature names expected by the model
            bin_edges: Bin edges for classification models
            preprocessing_params: Additional preprocessing parameters
        """
        self.model = model
        self.model_type = model_type
        self.task_type = task_type
        self.features = features
        self.bin_edges = bin_edges
        self.preprocessing_params = preprocessing_params or {}
        
        # For quantile ensemble models, extract internal columns if available
        if isinstance(model, QuantileEnsemblePredictor) and hasattr(model, 'columns'):
            self.encoded_features = model.columns
        else:
            self.encoded_features = None
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing specific to the model type.
        
        Args:
            X: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame ready for model prediction
        """
        X_processed = X.copy()
        
        if self.model_type == 'quantile_ensemble':
            # QE models handle their own preprocessing internally
            return X_processed
        else:
            # For classification models, apply one-hot encoding and feature alignment
            return self._preprocess_classification_features(X_processed)
    
    def _preprocess_classification_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing for classification models (LightGBM, XGBoost, etc.).
        
        Args:
            X: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame with one-hot encoding and aligned features
        """
        X_processed = X.copy()
        
        # Apply one-hot encoding for categorical features that the model expects
        categorical_features = ['location', 'component', 'makeType', 'bp_arch', 'bp_compiler', 'bp_opt']
        
        for cat_feature in categorical_features:
            if cat_feature in X_processed.columns:
                # Get dummies for the categorical feature
                dummies = pd.get_dummies(X_processed[cat_feature], prefix=cat_feature, dtype=int)
                
                # Add dummy columns to X_processed
                for dummy_col in dummies.columns:
                    X_processed[dummy_col] = dummies[dummy_col]
        
        # Add feature mapping for compatibility
        if 'lag_max_rss_g1_w1' in X_processed.columns and 'lag_max_rss_global_w5' not in X_processed.columns:
            X_processed['lag_max_rss_global_w5'] = X_processed['lag_max_rss_g1_w1']
        
        # Handle missing one-hot encoded features by creating them with zeros
        for feature in self.features:
            if feature not in X_processed.columns:
                # Check if it's a one-hot encoded categorical feature
                categorical_prefixes = ['location_', 'component_', 'makeType_', 'bp_arch_', 'bp_compiler_', 'bp_opt_']
                for cat_prefix in categorical_prefixes:
                    if feature.startswith(cat_prefix):
                        X_processed[feature] = 0
                        break
        
        # Select only the features the model expects and ensure correct order
        available_features = [f for f in self.features if f in X_processed.columns]
        
        if len(available_features) < len(self.features) * 0.5:  # Require at least 50% of features
            raise ValueError(f"Too few matching features: {len(available_features)}/{len(self.features)}")
        
        X_model = X_processed[self.features].copy()
        
        # Convert categorical columns to numeric if needed for prediction
        for col in X_model.columns:
            if X_model[col].dtype.name == 'category':
                X_model[col] = X_model[col].astype(float)
        
        X_model = X_model.fillna(0)
        return X_model
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.6) -> np.ndarray:
        """
        Make predictions on raw input data.
        
        Args:
            X: Raw input DataFrame (no preprocessing required)
            confidence_threshold: Confidence threshold for classification models
            
        Returns:
            Array of predictions (allocations for classification, direct values for regression)
        """
        # Preprocess the input data
        X_processed = self._preprocess_features(X)
        
        if self.task_type == 'classification':
            return self._predict_classification(X_processed, confidence_threshold)
        else:  # regression
            return self._predict_regression(X_processed)
    
    def _predict_classification(self, X_processed: pd.DataFrame, confidence_threshold: float) -> np.ndarray:
        """
        Make classification predictions and convert to memory allocations.
        
        Args:
            X_processed: Preprocessed input DataFrame
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Array of memory allocation predictions
        """
        if self.bin_edges is None:
            raise ValueError("Bin edges are required for classification models")
        
        # Make predictions
        y_pred_probs = self.model.predict_proba(X_processed)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # Apply confidence threshold adjustment
        num_classes = len(self.model.classes_)
        preds = []
        for i, probs in enumerate(y_pred_probs):
            pred = y_pred_classes[i]
            if probs[pred] < confidence_threshold:
                pred = min(pred + 1, num_classes - 1)
            preds.append(pred)
        
        # Convert class predictions to allocations using bin edges
        allocations = self.bin_edges[np.minimum(np.array(preds) + 1, len(self.bin_edges) - 1)]
        return allocations
    
    def _predict_regression(self, X_processed: pd.DataFrame) -> np.ndarray:
        """
        Make regression predictions.
        
        Args:
            X_processed: Preprocessed input DataFrame
            
        Returns:
            Array of regression predictions
        """
        return self.model.predict(X_processed)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the wrapped model.
        
        Returns:
            Dictionary containing model metadata
        """
        info = {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'num_features': len(self.features),
            'features': self.features.copy(),
            'preprocessing_params': self.preprocessing_params.copy()
        }
        
        if self.bin_edges is not None:
            info['bin_edges'] = self.bin_edges.tolist()
            info['num_classes'] = len(self.bin_edges) - 1
        
        if hasattr(self.model, 'get_params'):
            try:
                info['model_params'] = self.model.get_params()
            except Exception:
                # Some models may have issues with get_params, skip it
                pass
        
        return info
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the unified model wrapper to disk.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'UnifiedModelWrapper':
        """
        Load a unified model wrapper from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded UnifiedModelWrapper instance
        """
        return joblib.load(filepath)


def load_model(filepath: Union[str, Path]) -> UnifiedModelWrapper:
    """
    Load a UnifiedModelWrapper from disk.
    
    Args:
        filepath: Path to the unified model file
        
    Returns:
        UnifiedModelWrapper instance
        
    Raises:
        ValueError: If the file doesn't contain a UnifiedModelWrapper
    """
    filepath = Path(filepath)
    
    try:
        model = joblib.load(filepath)
        if isinstance(model, UnifiedModelWrapper):
            return model
        else:
            raise ValueError(f"File contains {type(model)}, expected UnifiedModelWrapper")
            
    except Exception as e:
        raise ValueError(f"Failed to load unified model from {filepath}: {e}")
    


