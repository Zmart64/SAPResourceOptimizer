"""
Configuration file
"""

import os


class Config:
    """Configuration class containing all project settings"""

    # Paths
    BASE_PATH = "/home/krebs/Distributed_Systems_Project/"
    DATA_DIR = os.path.join(BASE_PATH, "data")
    DATA_FILE = "build-data.csv"
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

    RESULTS_DIR = os.path.join(BASE_PATH, "results")
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

    # Data preprocessing parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    PRE_AVG_N = 3

    SIMILARITY_FEATURES = [
        "component",
        "compiler",
        "optimization",
        "jobs",
    ]
    # Model parameters
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    }

    XGBOOST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    }


config = Config()
