"""
Configuration file
"""

import os


class Config:
    """Configuration class containing all project settings"""

    # Paths
    BASE_PATH = "/home/wysokinska/Distributed_Systems_Project/"
    DATA_DIR = os.path.join(BASE_PATH, "sap_data")
    DATA_FILE = "build-data.csv"
    DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

    RESULTS_DIR = os.path.join(BASE_PATH, "results")
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

    # Data preprocessing parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    PRE_AVG_N = 3

    AVAILABLE_FEATURES = [
        "atom_id",
        "time",
        "location",
        "memory_fail_count",
        "branch",
        "buildProfile",
        "jobs",
        "localJobs",
        "makeType",
        "targets",
        "component",
        "ts_phase",
        "ts_status",
        "cgroup",
        "max_rss",
        "max_cache",
        "memreq",
    ]

    SIMILARITY_FEATURES = [
        "component",
        "buildProfile",
        "jobs",
    ]

    # Model parameters
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    XGBOOST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


config = Config()
