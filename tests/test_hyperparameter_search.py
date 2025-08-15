#!/usr/bin/env python3
"""
Test script to run hyperparameter search with minimal trials to verify everything works.
"""

import os
import sys
import warnings

from resource_prediction.config import Config
from resource_prediction.data_processing.preprocessor import DataPreprocessor
from resource_prediction.training.trainer import Trainer

# Set thread limits for parallel processing
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[var] = "1"

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class TestConfig(Config):
    """Modified configuration for testing with minimal trials."""
    
    # Override to run only 1 trial per family for testing
    N_CALLS_PER_FAMILY = 1
    
    # Use fewer parallel workers for testing
    NUM_PARALLEL_WORKERS = 1
    
    # Use fewer CV splits for faster execution
    CV_SPLITS = 2
    
    # Use the correct data file name that exists
    RAW_DATA_PATH = Config.DATA_DIR / "raw" / "build-data-sorted.csv"
    
    # Ensure outputs go to artifacts directory
    OUTPUT_DIR = Config.PROJECT_ROOT / "artifacts"


def main():
    """Run a minimal hyperparameter search to test functionality."""
    print("=" * 60)
    print("HYPERPARAMETER SEARCH TEST")
    print("=" * 60)
    print("Running with 1 trial per model family for testing purposes")
    print("CV splits: 2 (reduced for faster execution)")
    print("Parallel workers: 1")
    print()
    
    config = TestConfig()
    
    # Check if processed data exists
    if not all([
        config.X_TRAIN_PATH.exists(),
        config.Y_TRAIN_PATH.exists(),
        config.X_TEST_PATH.exists(),
        config.Y_TEST_PATH.exists()
    ]):
        print("Processed data not found. Running preprocessing first...")
        preprocessor = DataPreprocessor(config)
        preprocessor.process()
        print("Preprocessing complete.\n")
    else:
        print("Found existing processed data. Skipping preprocessing.\n")
    
    # Run hyperparameter search with minimal trials
    print("Starting hyperparameter search...")
    trainer = Trainer(
        config,
        evaluate_all_archs=True,  # Test all architectures
        task_type_filter=None,   # Test both regression and classification
        save_models=False        # Don't save models for testing
    )
    
    try:
        trainer.run_optimization_and_evaluation()
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ All model families were tested")
        print("✅ Evaluation pipeline worked correctly")
        print("✅ No critical errors encountered")
        
        # Show results locations
        print("\nResults saved to:")
        if config.REGRESSION_RESULTS_CSV_PATH.exists():
            print(f"  📊 Regression results: {config.REGRESSION_RESULTS_CSV_PATH}")
        if config.CLASSIFICATION_RESULTS_CSV_PATH.exists():
            print(f"  📊 Classification results: {config.CLASSIFICATION_RESULTS_CSV_PATH}")
        if config.ALLOCATION_PLOT_PATH.exists():
            print(f"  📈 Allocation plot: {config.ALLOCATION_PLOT_PATH}")
        if config.RESULTS_PLOT_PATH.exists():
            print(f"  📈 Comparison plot: {config.RESULTS_PLOT_PATH}")
            
    except Exception as e:
        print(f"\n❌ ERROR during hyperparameter search: {str(e)}")
        print("This indicates an issue that needs to be resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
