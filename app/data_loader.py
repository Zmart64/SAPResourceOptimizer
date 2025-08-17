"""
Unified data loader for simulation across all model types.

This module provides a unified interface to load holdout simulation data
that works consistently across classification and regression models.
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from resource_prediction.config import Config
from resource_prediction.data_processing.preprocessor import DataPreprocessor


class UnifiedDataLoader:
    """Loads simulation data consistently for all model types."""
    
    def __init__(self):
        self.config = Config()
        
    def load_simulation_data(self):
        """
        Load holdout simulation data for all model types.
        
        Returns:
            pd.DataFrame: Simulation dataframe with features, targets, and metadata
                         Required columns: time, actual memory columns, memreq_gb, all features
        """
        # Check if processed data exists, if not create it
        if not self._processed_data_exists():
            print("Processed data not found. Running preprocessing...")
            preprocessor = DataPreprocessor(self.config)
            preprocessor.process()
        
        # Load the holdout test data
        try:
            X_test = pd.read_pickle(self.config.X_TEST_PATH)
            y_test = pd.read_pickle(self.config.Y_TEST_PATH)
            
            # We need to get the time and memreq_gb columns from the raw data
            # Load the raw data and extract the test portion
            df_raw = pd.read_csv(self.config.RAW_DATA_PATH, sep=";")
            
            # Apply the same preprocessing steps to get the time and memreq columns
            df_raw['time'] = pd.to_datetime(df_raw['time'], format='mixed', errors='coerce')
            df_raw.dropna(subset=['time'], inplace=True)
            
            # Apply the same transformations as in preprocessor
            df_raw["max_rss_gb"] = df_raw["max_rss"] / 1e9
            df_raw["memreq_gb"] = df_raw["memreq"] / 1e9
            
            # Get the test split (same logic as preprocessor)
            split_index = int(len(df_raw) * (1 - self.config.TEST_SET_FRACTION))
            df_test_raw = df_raw.iloc[split_index:].reset_index(drop=True)
            
            # Combine the processed features with raw metadata
            simulation_df = X_test.reset_index(drop=True)
            simulation_df['time'] = df_test_raw['time'].reset_index(drop=True)
            simulation_df[self.config.TARGET_COLUMN_PROCESSED] = y_test[self.config.TARGET_COLUMN_PROCESSED].reset_index(drop=True)
            simulation_df['memreq_gb'] = df_test_raw['memreq_gb'].reset_index(drop=True)
            
            # Add categorical dtype conversion for streamlit compatibility
            categorical_cols = [
                "bp_arch", "bp_compiler", "bp_opt", "component", "makeType",
                "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_weekofyear"
            ]
            for col in categorical_cols:
                if col in simulation_df.columns:
                    simulation_df[col] = simulation_df[col].astype("category")
            
            # Sort by time for realistic simulation
            simulation_df = simulation_df.sort_values('time').reset_index(drop=True)
            
            print(f"Loaded simulation data: {len(simulation_df)} rows")
            print(f"Date range: {simulation_df['time'].min()} to {simulation_df['time'].max()}")
            
            return simulation_df
            
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return None
    
    def _processed_data_exists(self):
        """Check if all required processed data files exist."""
        required_files = [
            self.config.X_TRAIN_PATH,
            self.config.Y_TRAIN_PATH, 
            self.config.X_TEST_PATH,
            self.config.Y_TEST_PATH
        ]
        return all(path.exists() for path in required_files)
    
    def get_target_columns(self):
        """Return the appropriate target column names for different model types."""
        return {
            'actual_col': self.config.TARGET_COLUMN_PROCESSED,  # 'max_rss_gb'
            'memreq_col': 'memreq_gb'
        }


def load_unified_simulation_data():
    """Convenience function to load simulation data."""
    loader = UnifiedDataLoader()
    return loader.load_simulation_data()


def get_target_columns():
    """Convenience function to get target column names."""
    loader = UnifiedDataLoader()
    return loader.get_target_columns()