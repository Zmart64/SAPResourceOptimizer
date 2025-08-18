"""Preprocessing pipeline for consistent feature engineering across training and inference."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class ModelPreprocessor(BaseEstimator, TransformerMixin):
    """
    Handles feature preprocessing for model deployment.
    
    This preprocessor encapsulates all feature engineering logic that needs to be
    consistent between training and inference, including:
    - Categorical encoding (one-hot)
    - Feature alignment and ordering
    - Missing value handling
    - Feature mapping for backward compatibility
    """
    
    def __init__(
        self, 
        categorical_features: Optional[List[str]] = None,
        expected_features: Optional[List[str]] = None,
        feature_mappings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the preprocessor.
        
        Args:
            categorical_features: List of categorical columns to encode
            expected_features: Expected feature names in the correct order
            feature_mappings: Dictionary mapping old feature names to new ones
        """
        self.categorical_features = categorical_features or [
            'location', 'component', 'makeType', 'bp_arch', 'bp_compiler', 'bp_opt'
        ]
        self.expected_features = expected_features or []
        self.feature_mappings = feature_mappings or {
            'lag_max_rss_g1_w1': 'lag_max_rss_global_w5'
        }
        
        # Will be populated during fit
        self.encoded_feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the preprocessor to training data.
        
        Args:
            X: Training features
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            self
        """
        X_copy = X.copy()
        
        # Apply feature mappings
        for old_name, new_name in self.feature_mappings.items():
            if old_name in X_copy.columns and new_name not in X_copy.columns:
                X_copy[new_name] = X_copy[old_name]
        
        # Generate one-hot encoded features to determine expected columns
        X_encoded = self._apply_categorical_encoding(X_copy)
        
        # Store the feature names and order from training
        self.encoded_feature_names_ = X_encoded.columns.tolist()
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_copy = X.copy()
        
        # Apply feature mappings
        for old_name, new_name in self.feature_mappings.items():
            if old_name in X_copy.columns and new_name not in X_copy.columns:
                X_copy[new_name] = X_copy[old_name]
        
        # Apply categorical encoding
        X_encoded = self._apply_categorical_encoding(X_copy)
        
        # Align features with training data
        X_aligned = self._align_features(X_encoded)
        
        return X_aligned
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _apply_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding to categorical features."""
        X_processed = X.copy()
        
        for cat_feature in self.categorical_features:
            if cat_feature in X_processed.columns:
                # Get dummies for the categorical feature
                dummies = pd.get_dummies(
                    X_processed[cat_feature], 
                    prefix=cat_feature, 
                    dtype=int
                )
                
                # Add dummy columns to X_processed
                for dummy_col in dummies.columns:
                    X_processed[dummy_col] = dummies[dummy_col]
        
        return X_processed
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align features with the training feature set."""
        X_aligned = X.copy()
        
        # Add missing features with zeros
        for feature in self.encoded_feature_names_:
            if feature not in X_aligned.columns:
                # Check if it's a categorical dummy variable
                is_categorical_dummy = any(
                    feature.startswith(f"{cat}_") 
                    for cat in self.categorical_features
                )
                if is_categorical_dummy:
                    X_aligned[feature] = 0
        
        # Select only the expected features in the correct order
        available_features = [f for f in self.encoded_feature_names_ if f in X_aligned.columns]
        
        if len(available_features) < len(self.encoded_feature_names_) * 0.5:
            raise ValueError(
                f"Too few matching features: {len(available_features)}/{len(self.encoded_feature_names_)}"
            )
        
        # Reorder and select features
        X_final = X_aligned[self.encoded_feature_names_].copy()
        
        # Convert categorical columns to numeric and handle missing values
        for col in X_final.columns:
            if X_final[col].dtype.name == 'category':
                X_final[col] = X_final[col].astype(float)
        
        X_final = X_final.fillna(0)
        
        return X_final
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names for sklearn compatibility."""
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        return self.encoded_feature_names_
