"""Project configuration and hyperparameter search spaces."""

from pathlib import Path

import optuna

# Direct model imports following Zmart's pattern
from resource_prediction.models import (
    GBLGBQuantileEnsemble,
    GBXGBQuantileEnsemble,
    LGBCatQuantileEnsemble,
    LGBXGBQuantileEnsemble,
    LightGBMClassifier,
    LightGBMRegressor,
    LogisticRegression,
    RandomForestClassifier,
    SizeyPredictor,
    XGBCatQuantileEnsemble,
    XGBoostClassifier,
    XGBoostRegressor,
    XGBXGBQuantileEnsemble,
)
from resource_prediction.models.implementations.sizey import (
    OffsetStrategy,
    UnderPredictionStrategy,
)


class Config:
    """
    Central configuration class for the resource prediction project.

    This class holds all static configuration values, including file paths,
    feature definitions, and hyperparameter search spaces for all models.
    """

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
    MODELS_DIR = PROJECT_ROOT / "artifacts" / "trained_models"
    ALLOCATION_PLOT_PATH = OUTPUT_DIR / "memory_allocation_plot.png"
    ALLOCATION_SUMMARY_REPORT_PATH = OUTPUT_DIR / "allocation_summary_report.csv"

    RAW_DATA_PATH = DATA_DIR / "raw" / "build-data-4.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    BASELINE_STATS_PATH = PROCESSED_DATA_DIR / "baseline_allocation_stats.pkl"
    X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.pkl"
    Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
    X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
    Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"

    OPTUNA_DB_DIR = OUTPUT_DIR / "optuna_db"
    REGRESSION_RESULTS_CSV_PATH = OUTPUT_DIR / "regression_results.csv"
    CLASSIFICATION_RESULTS_CSV_PATH = OUTPUT_DIR / "classification_results.csv"
    RESULTS_PLOT_PATH = OUTPUT_DIR / "comparison_chart.png"
    # Path for holdout score vs prediction time scatter plot
    SCORE_TIME_PLOT_PATH = OUTPUT_DIR / "score_vs_prediction_time.png"

    TARGET_COLUMN_RAW = "max_rss"
    TARGET_COLUMN_PROCESSED = "max_rss_gb"
    TEST_SET_FRACTION = 0.10
    RANDOM_STATE = 42

    BASE_FEATURES = [
        "location",
        "component",
        "makeType",
        "bp_arch",
        "bp_compiler",
        "bp_opt",
        "ts_year",
        "ts_month",
        "ts_dow",
        "ts_hour",
        "ts_weekofyear",
        "branch_prefix",
        "jobs",
        "localJobs",
        "target_cnt",
        "target_has_dist",
        "branch_id_str",
        "lag_1_grouped",
        "lag_max_rss_global_w5",
        "rolling_p95_rss_g1_w5",
    ]
    ALL_FEATURES = list(dict.fromkeys(BASE_FEATURES))

    CV_SPLITS = 3
    N_CALLS_PER_FAMILY = 128
    NUM_PARALLEL_WORKERS = 8

    MODEL_FAMILIES = {
        # Standard QE ensemble: GradientBoosting + XGBoost (formerly qe_regression)
        "gb_xgb_ensemble": {
            "type": "regression",
            "base_model": "gb_xgb_quantile_ensemble",
            "class": GBXGBQuantileEnsemble,
        },
        "lgb_xgb_ensemble": {
            "type": "regression",
            "base_model": "lgb_xgb_quantile_ensemble",
            "class": LGBXGBQuantileEnsemble,
        },
        "gb_lgb_ensemble": {
            "type": "regression",
            "base_model": "gb_lgb_quantile_ensemble",
            "class": GBLGBQuantileEnsemble,
        },
        "xgb_cat_ensemble": {
            "type": "regression",
            "base_model": "xgb_cat_quantile_ensemble",
            "class": XGBCatQuantileEnsemble,
        },
        "lgb_cat_ensemble": {
            "type": "regression",
            "base_model": "lgb_cat_quantile_ensemble",
            "class": LGBCatQuantileEnsemble,
        },
        "xgb_xgb_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_quantile_ensemble",
            "class": XGBXGBQuantileEnsemble,
        },
        "xgboost_classification": {
            "type": "classification",
            "base_model": "xgboost",
            "class": XGBoostClassifier,
        },
        "xgboost_regression": {
            "type": "regression",
            "base_model": "xgboost",
            "class": XGBoostRegressor,
        },
        "lightgbm_classification": {
            "type": "classification",
            "base_model": "lightgbm",
            "class": LightGBMClassifier,
        },
        "rf_classification": {
            "type": "classification",
            "base_model": "random_forest",
            "class": RandomForestClassifier,
        },
        "lightgbm_regression": {
            "type": "regression",
            "base_model": "lightgbm",
            "class": LightGBMRegressor,
        },
        "lr_classification": {
            "type": "classification",
            "base_model": "logistic_regression",
            "class": LogisticRegression,
        },
        "sizey_regression": {
            "type": "regression",
            "base_model": "sizey",
            "class": SizeyPredictor,
        },
    }

    # Hyperparameter configuration system
    # Each model family defines ONLY the parameters it actually needs
    # No shared parameters, no cross-contamination between models
    HYPERPARAMETER_CONFIGS = {
        # GradientBoosting + XGBoost Ensemble for Quantile Regression (standard QE)
        "gb_xgb_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "gb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "gb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "gb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
            "xgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "xgb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "xgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
        },
        # LightGBM + XGBoost Ensemble
        "lgb_xgb_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "lgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "lgb_num_leaves": {"min": 15, "max": 100, "type": "int", "default": 31},
            "lgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
            "xgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "xgb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "xgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
        },
        # GradientBoosting + LightGBM Ensemble
        "gb_lgb_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "gb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "gb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "gb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
            "lgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "lgb_num_leaves": {"min": 15, "max": 100, "type": "int", "default": 31},
            "lgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
        },
        # XGBoost + CatBoost Ensemble
        "xgb_cat_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "xgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "xgb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "xgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
            "cat_iterations": {"min": 200, "max": 700, "type": "int", "default": 300},
            "cat_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "cat_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
        },
        # LightGBM + CatBoost Ensemble
        "lgb_cat_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "lgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "lgb_num_leaves": {"min": 15, "max": 100, "type": "int", "default": 31},
            "lgb_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
            "cat_iterations": {"min": 200, "max": 700, "type": "int", "default": 300},
            "cat_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "cat_lr": {
                "min": 0.01,
                "max": 0.15,
                "type": "float",
                "log": True,
                "default": 0.05,
            },
        },
        # XGBoost + XGBoost Ensemble (standard ranges like other ensembles)
        "xgb_xgb_ensemble": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # First XGBoost model parameters
            "xgb1_n_estimators": {
                "min": 200,
                "max": 800,
                "type": "int",
                "default": 400,
            },
            "xgb1_max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
            "xgb1_lr": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
            # Second XGBoost model parameters
            "xgb2_n_estimators": {
                "min": 200,
                "max": 800,
                "type": "int",
                "default": 400,
            },
            "xgb2_max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
            "xgb2_lr": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
        },
        # XGBoost Regression - only XGBoost regression parameters
        "xgboost_regression": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
            "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
            "learning_rate": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
        },
        # XGBoost Classification - only XGBoost classification parameters
        "xgboost_classification": {
            "confidence_threshold": {
                "min": 0.3,
                "max": 0.9,
                "type": "float",
                "default": 0.5,
            },
            "n_bins": {"min": 3, "max": 15, "type": "int", "default": 7},
            "strategy": {
                "choices": ["uniform", "quantile", "kmeans"],
                "default": "uniform",
            },
            "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
            "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
            "learning_rate": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
        },
        # LightGBM Regression - only LightGBM regression parameters
        "lightgbm_regression": {
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "n_estimators": {"min": 100, "max": 700, "type": "int", "default": 400},
            "num_leaves": {"min": 20, "max": 60, "type": "int", "default": 31},
            "learning_rate": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
            # max_depth is optional for LightGBM, not included by default
        },
        # LightGBM Classification - only LightGBM classification parameters
        "lightgbm_classification": {
            "confidence_threshold": {
                "min": 0.3,
                "max": 0.9,
                "type": "float",
                "default": 0.5,
            },
            "n_bins": {"min": 3, "max": 15, "type": "int", "default": 7},
            "strategy": {
                "choices": ["uniform", "quantile", "kmeans"],
                "default": "uniform",
            },
            "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
            "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
            "num_leaves": {"min": 20, "max": 64, "type": "int", "default": 31},
            "learning_rate": {
                "min": 0.01,
                "max": 0.2,
                "type": "float",
                "log": True,
                "default": 0.1,
            },
        },
        # Random Forest Classification - only RF parameters (no alpha, no learning_rate)
        "rf_classification": {
            "confidence_threshold": {
                "min": 0.3,
                "max": 0.9,
                "type": "float",
                "default": 0.5,
            },
            "n_bins": {"min": 3, "max": 15, "type": "int", "default": 7},
            "strategy": {
                "choices": ["uniform", "quantile", "kmeans"],
                "default": "uniform",
            },
            "n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "max_depth": {"min": 6, "max": 15, "type": "int", "default": 10},
        },
        # Logistic Regression Classification - only LR parameters
        "lr_classification": {
            "confidence_threshold": {
                "min": 0.3,
                "max": 0.9,
                "type": "float",
                "default": 0.5,
            },
            "n_bins": {"min": 3, "max": 15, "type": "int", "default": 7},
            "strategy": {
                "choices": ["uniform", "quantile", "kmeans"],
                "default": "uniform",
            },
            "C": {
                "min": 1e-2,
                "max": 10.0,
                "type": "float",
                "log": True,
                "default": 1.0,
            },
            "solver": {"choices": ["liblinear", "saga"], "default": "liblinear"},
            "penalty": {"choices": ["l1", "l2", "elasticnet"], "default": "l2"},
            "l1_ratio": {
                "min": 0,
                "max": 1,
                "type": "float",
                "default": 0.5,
            },  # Only used with elasticnet
        },
        # Sizey Regression - Sizey-specific parameters
        "sizey_regression": {
            "alpha": {"min": 0.01, "max": 0.5, "type": "float", "default": 0.1},
            "beta": {"min": 0.0, "max": 1.0, "type": "float", "default": 1.0},
            "offset_strat": {
                "choices": ["DYNAMIC", "STD_ALL", "MED_UNDER", "MED_ALL", "STD_UNDER"],
                "default": "DYNAMIC",
            },
            "error_strat": {
                "choices": ["MAX_EVER_OBSERVED", "DOUBLE"],
                "default": "MAX_EVER_OBSERVED",
            },
            "use_softmax": {"choices": [True, False], "default": True},
            "error_metric": {
                "choices": ["smoothed_mape", "neg_mean_squared_error"],
                "default": "smoothed_mape",
            },
        },
    }

    @staticmethod
    def _suggest_param(trial, param_name, param_config):
        """
        Helper method to generate optuna suggestion based on parameter configuration.

        Args:
            trial: Optuna trial object
            param_name (str): Name of the parameter
            param_config (dict): Configuration for the parameter

        Returns:
            Suggested parameter value
        """
        if "choices" in param_config:
            return trial.suggest_categorical(param_name, param_config["choices"])
        elif param_config.get("type") == "int":
            return trial.suggest_int(
                param_name, param_config["min"], param_config["max"]
            )
        elif param_config.get("type") == "float":
            log = param_config.get("log", False)
            return trial.suggest_float(
                param_name, param_config["min"], param_config["max"], log=log
            )
        else:
            raise ValueError(
                f"Invalid parameter configuration for {param_name}: {param_config}"
            )

    @staticmethod
    def _apply_family_specific_transformations(params, family_name):
        """Applies model-specific transformations, like setting objectives."""
        if family_name == "xgboost_regression":
            params["objective"] = "reg:quantileerror"
            if "alpha" in params:
                params["quantile_alpha"] = params.pop("alpha")
        elif family_name == "lightgbm_regression":
            params["objective"] = "quantile"
        elif family_name == "lr_classification":
            if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                params.pop("l1_ratio")
            # Pruning for incompatible solver/penalty combinations
            if (
                params.get("solver") == "liblinear"
                and params.get("penalty") == "elasticnet"
            ) or (
                params.get("penalty") == "elasticnet" and params.get("solver") != "saga"
            ):
                raise optuna.exceptions.TrialPruned()
        elif family_name == "sizey_regression":
            # Convert string enum values back to actual enum objects
            if "offset_strat" in params:
                offset_str = params["offset_strat"]
                params["offset_strat"] = getattr(OffsetStrategy, offset_str)
            if "error_strat" in params:
                error_str = params["error_strat"]
                params["error_strat"] = getattr(UnderPredictionStrategy, error_str)
        return params

    @staticmethod
    def get_search_space(trial, family_name):
        """
        Defines the hyperparameter search space for a given model family.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.
            family_name (str): The model family name.

        Returns:
            dict: A dictionary of suggested hyperparameters for the trial.
        """
        params = {}
        family_config = Config.HYPERPARAMETER_CONFIGS.get(family_name, {})
        if not family_config:
            raise ValueError(f"No config for model family '{family_name}'")

        for param, config in family_config.items():
            params[param] = Config._suggest_param(trial, param, config)

        params = Config._apply_family_specific_transformations(params, family_name)

        return params

    @staticmethod
    def get_defaults(family_name):
        """
        Get default hyperparameters for a given model family.

        Args:
            family_name (str): The model family name.

        Returns:
            dict: A dictionary of default hyperparameters.
        """
        params = {}
        family_config = Config.HYPERPARAMETER_CONFIGS.get(family_name, {})
        if not family_config:
            raise ValueError(f"No config for model family '{family_name}'")

        for param, config in family_config.items():
            if "default" in config:
                params[param] = config["default"]
            else:
                raise ValueError(f"No default for '{param}' in '{family_name}'")

        return Config._apply_family_specific_transformations(params, family_name)
