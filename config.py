# config.py

import numpy as np


class Config:
    """Configuration class for the training pipeline"""

    # --- Files and Directories ---
    # The full, unsplit dataset
    FULL_DATA_FILE = "build-data-sorted.csv"
    OUTPUT_DIR = "output_model_and_plots"
    MODEL_FILENAME = "xgb_classifier_model.pkl"

    # --- Data Splitting ---
    # The proportion of data to be used for training. The rest is held out for simulation.
    TRAIN_DATA_RATIO = 0.8

    # --- Target Variable and Binning ---
    TARGET_COLUMN = "max_rss"
    # Manually defined bin edges in GB.
    MANUAL_BIN_EDGES_GB = sorted(
        list(set([3.00e-02, 5.77e+01, 1.15e+02, 1.73e+02, 2.31e+02]))
    )

    # --- Cross-Validation, Grid Search, and Model Params ---
    N_SPLITS_CV = 3
    GRID_SEARCH_N_JOBS = -1  # Use all available cores
    RANDOM_STATE = 42

    # Scorer weights for different classes during GridSearch
    RECALL_WEIGHTS = np.array([0.05, 0.15, 0.30, 0.50])

    # Class weights for training to penalize misclassification of high-memory jobs
    TRAINING_CLASS_WEIGHTS_EXPONENT = 1.5

    # Hyperparameter Grid for XGBoost Classifier
    PARAM_GRID = {
        'n_estimators': [200, 400],
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [5, 6, 7],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
    }
