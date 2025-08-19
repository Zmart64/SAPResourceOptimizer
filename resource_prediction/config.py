"""Project configuration and hyperparameter search spaces."""

from pathlib import Path
import importlib

import optuna


def _import_model_class(module_path: str, class_name: str):
    """Dynamically import a model class to avoid manual imports."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_name} from {module_path}: {e}")


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
    QUANT_FEATURES = [
        "build_load",
        "target_intensity",
        "debug_multiplier",
        "heavy_target_flag",
        "high_parallelism",
    ]
    ALL_FEATURES = list(dict.fromkeys(BASE_FEATURES + QUANT_FEATURES))

    CV_SPLITS = 3
    N_CALLS_PER_FAMILY = 10
    NUM_PARALLEL_WORKERS = 1

    MODEL_FAMILIES = {
        "qe_regression": {
            "type": "regression",
            "base_model": "quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "QuantileEnsemblePredictor"),
        },
        "lgb_xgb_ensemble": {
            "type": "regression",
            "base_model": "lgb_xgb_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "LGBXGBQuantileEnsemble"),
        },
        "gb_lgb_ensemble": {
            "type": "regression",
            "base_model": "gb_lgb_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "GBLGBQuantileEnsemble"),
        },
        "xgb_cat_ensemble": {
            "type": "regression",
            "base_model": "xgb_cat_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBCatQuantileEnsemble"),
        },
        "lgb_cat_ensemble": {
            "type": "regression",
            "base_model": "lgb_cat_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "LGBCatQuantileEnsemble"),
        },
        "xgb_xgb_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBXGBQuantileEnsemble"),
        },
        "xgb_xgb_max_ensemble": {
            "type": "regression", 
            "base_model": "xgb_xgb_max_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBXGBMaxQuantileEnsemble"),
        },
        "xgb_xgb_weighted_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_weighted_quantile_ensemble", 
            "class": _import_model_class("resource_prediction.models", "XGBXGBWeightedQuantileEnsemble"),
        },
        "xgb_xgb_confidence_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_confidence_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBXGBConfidenceEnsemble"),
        },
        "xgb_xgb_adaptive_safety_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_adaptive_safety_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBXGBAdaptiveSafetyEnsemble"),
        },
        "xgb_xgb_selective_ensemble": {
            "type": "regression",
            "base_model": "xgb_xgb_selective_quantile_ensemble",
            "class": _import_model_class("resource_prediction.models", "XGBXGBSelectiveEnsemble"),
        },
        "xgboost_classification": {
            "type": "classification",
            "base_model": "xgboost",
            "class": _import_model_class("resource_prediction.models", "XGBoostClassifier"),
        },
        "xgboost_regression": {
            "type": "regression",
            "base_model": "xgboost",
            "class": _import_model_class("resource_prediction.models", "XGBoostRegressor"),
        },
        "lightgbm_classification": {
            "type": "classification",
            "base_model": "lightgbm",
            "class": _import_model_class("resource_prediction.models", "LightGBMClassifier"),
        },
        "rf_classification": {
            "type": "classification",
            "base_model": "random_forest",
            "class": _import_model_class("resource_prediction.models", "RandomForestClassifier"),
        },
        "lightgbm_regression": {
            "type": "regression",
            "base_model": "lightgbm",
            "class": _import_model_class("resource_prediction.models", "LightGBMRegressor"),
        },
        "lr_classification": {
            "type": "classification",
            "base_model": "logistic_regression",
            "class": _import_model_class("resource_prediction.models", "LogisticRegression"),
        },
        "sizey_regression": {
            "type": "regression",
            "base_model": "sizey",
            "class": _import_model_class("resource_prediction.models", "SizeyPredictor"),
        },
    }

    # Hyperparameter configuration system
    # Each model family defines ONLY the parameters it actually needs
    # No shared parameters, no cross-contamination between models
    HYPERPARAMETER_CONFIGS = {
        # Quantile Ensemble Regression - only QE-specific parameters
        "qe_regression": {
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
        # XGBoost + XGBoost Specialized Ensemble
        "xgb_xgb_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost + XGBoost Max Ensemble (traditional maximum selection)
        "xgb_xgb_max_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost + XGBoost Weighted Ensemble (intelligent routing)
        "xgb_xgb_weighted_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost + XGBoost Confidence Ensemble (confidence-based selection)
        "xgb_xgb_confidence_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost + XGBoost Adaptive Safety Ensemble (adaptive safety factors)
        "xgb_xgb_adaptive_safety_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost + XGBoost Selective Ensemble (selective conservative usage)
        "xgb_xgb_selective_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            # Conservative model parameters (higher quantile, deeper trees, fewer estimators)
            "conservative_quantile": {"choices": [0.95, 0.98, 0.99], "default": 0.98},
            "conservative_n_estimators": {"min": 100, "max": 400, "type": "int", "default": 200},
            "conservative_max_depth": {"min": 6, "max": 12, "type": "int", "default": 8},
            "conservative_lr": {
                "min": 0.01,
                "max": 0.10,
                "type": "float",
                "log": True,
                "default": 0.03,
            },
            # Aggressive model parameters (lower quantile, shallower trees, more estimators)
            "aggressive_quantile": {"choices": [0.85, 0.90, 0.95], "default": 0.90},
            "aggressive_n_estimators": {"min": 300, "max": 800, "type": "int", "default": 500},
            "aggressive_max_depth": {"min": 3, "max": 7, "type": "int", "default": 5},
            "aggressive_lr": {
                "min": 0.03,
                "max": 0.20,
                "type": "float",
                "log": True,
                "default": 0.08,
            },
        },
        # XGBoost Regression - only XGBoost regression parameters
        "xgboost_regression": {
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
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
            "use_quant_feats": {"choices": [True, False], "default": True},
            "sizey_alpha": {"min": 0.01, "max": 0.5, "type": "float", "default": 0.1},
            "offset_strat": {
                "choices": ["DYNAMIC", "STD", "MED_UNDER", "MED_ALL", "STDUNDER"],
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
    def get_default_params(base_model, task_type):
        """
        Get default parameter values for a model without hyperparameter search.

        Args:
            base_model (str): The base algorithm name (e.g., 'xgboost').
            task_type (str): The task type ('regression' or 'classification').

        Returns:
            dict: A dictionary of default parameter values.
        """
        params = {}

        # Add task-specific common parameters (currently only for classification)
        task_common_key = f"{task_type}_common"
        for param, config in Config.HYPERPARAMETER_CONFIGS.get(
            task_common_key, {}
        ).items():
            if "default" in config:
                params[param] = config["default"]
            elif "choices" in config:
                params[param] = config["choices"][0]

        # Add model-specific parameters
        model_config = Config.HYPERPARAMETER_CONFIGS.get(base_model, {})
        if task_type in model_config:
            # Model has task-specific config
            model_params = model_config[task_type]
        else:
            # Model uses same config for all tasks
            model_params = model_config

        for param, config in model_params.items():
            if "default" in config:
                params[param] = config["default"]
            elif "choices" in config:
                params[param] = config["choices"][0]

        return params

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
    def get_search_space(trial, family_name):
        """
        Defines the hyperparameter search space for a given model family.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.
            family_name (str): The model family name (e.g., 'xgboost_regression', 'lightgbm_classification').

        Returns:
            dict: A dictionary of suggested hyperparameters for the trial.
        """
        params = {}

        # Get parameters for the specific model family
        family_config = Config.HYPERPARAMETER_CONFIGS.get(family_name, {})
        if not family_config:
            raise ValueError(
                f"No hyperparameter configuration found for model family '{family_name}'"
            )

        for param, config in family_config.items():
            params[param] = Config._suggest_param(trial, param, config)

        # Add model-specific transformations for internal parameter names
        if family_name == "xgboost_regression":
            # For XGBoost regression, set specific objective and transform alpha to quantile_alpha
            params["objective"] = "reg:quantileerror"
            if "alpha" in params:
                params["quantile_alpha"] = params.pop("alpha")

        elif family_name == "lightgbm_regression":
            # For LightGBM regression, set objective to quantile
            params["objective"] = "quantile"

        elif family_name == "lr_classification":
            # Handle special constraint for logistic regression
            if (
                params.get("solver") == "liblinear"
                and params.get("penalty") == "elasticnet"
            ):
                raise optuna.exceptions.TrialPruned()
            if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                # Remove l1_ratio if penalty is not elasticnet
                params.pop("l1_ratio")
            elif (
                params.get("penalty") == "elasticnet" and params.get("solver") != "saga"
            ):
                raise optuna.exceptions.TrialPruned()

        return params

    @staticmethod
    def get_defaults(family_name):
        """
        Get default hyperparameters for a given model family.

        Args:
            family_name (str): The model family name (e.g., 'xgboost_regression').

        Returns:
            dict: A dictionary of default hyperparameters.
        """
        params = {}

        # Get parameters for the specific model family
        family_config = Config.HYPERPARAMETER_CONFIGS.get(family_name, {})
        if not family_config:
            raise ValueError(
                f"No hyperparameter configuration found for model family '{family_name}'"
            )

        for param, config in family_config.items():
            if "default" in config:
                params[param] = config["default"]
            else:
                raise ValueError(
                    f"No default value specified for parameter '{param}' in family '{family_name}'"
                )

        # Apply the same transformations as in get_search_space
        if family_name == "xgboost_regression":
            params["objective"] = "reg:quantileerror"
            if "alpha" in params:
                params["quantile_alpha"] = params.pop("alpha")

        elif family_name == "lightgbm_regression":
            params["objective"] = "quantile"

        elif family_name == "lr_classification":
            # Apply logistic regression constraints for defaults
            if params.get("penalty") != "elasticnet" and "l1_ratio" in params:
                params.pop("l1_ratio")

        return params
