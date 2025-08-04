from pathlib import Path
from skopt.space import Real, Integer, Categorical


class Config:
    """Main configuration class for the project."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "build-data-sorted.csv"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "features.pkl"
    PROCESSED_TARGET_PATH = PROCESSED_DATA_DIR / "target.pkl"

    # Paths for train/test splits
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"

    OUTPUT_DIR = PROJECT_ROOT / "output_classification"
    OPTIMIZATION_SUMMARY_PATH = OUTPUT_DIR / "optimization_summary.csv"

    # Data and Feature Engineering
    TARGET_COLUMN_RAW = "max_rss"
    # Target used for binning
    TARGET_COLUMN_PROCESSED = "max_rss_gb"

    # Training & Optimization Parameters
    TEST_SET_FRACTION = 0.2
    N_SPLITS_CV = 3
    MODELS_TO_OPTIMIZE = ['catboost', 'lightgbm', 'xgboost']
    # Increase for more thorough search (> 5)
    N_OPTIMIZATION_CALLS_PER_MODEL = 10
    RANDOM_STATE = 42

    # Bayesian Optimization Search Spaces (Classification)
    SEARCH_SPACES = {
        'catboost': [
            Integer(4, 10, name='n_bins'),
            Categorical(['quantile', 'uniform', 'kmeans'], name='strategy'),
            Real(0.1, 0.95, name='confidence_threshold'),
            Real(0.01, 0.2, name='learning_rate', prior='log-uniform'),
            Integer(4, 10, name='depth'),
            Integer(200, 800, name='iterations')
        ],
        'lightgbm': [
            Integer(4, 10, name='n_bins'),
            Categorical(['quantile', 'uniform', 'kmeans'], name='strategy'),
            Real(0.1, 0.95, name='confidence_threshold'),
            Real(0.01, 0.2, name='learning_rate', prior='log-uniform'),
            Integer(4, 10, name='max_depth'),
            Integer(20, 64, name='num_leaves'),
            Integer(200, 800, name='n_estimators')
        ],
        'xgboost': [
            Integer(4, 10, name='n_bins'),
            Categorical(['quantile', 'uniform', 'kmeans'], name='strategy'),
            Real(0.1, 0.95, name='confidence_threshold'),
            Real(0.01, 0.2, name='learning_rate', prior='log-uniform'),
            Integer(4, 10, name='max_depth'),
            Real(0.7, 1.0, name='subsample'),
            Integer(200, 800, name='n_estimators')
        ]
    }
