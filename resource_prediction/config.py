"""Project configuration and hyperparameter search spaces."""

from pathlib import Path
import optuna


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

    RAW_DATA_PATH = DATA_DIR / "raw" / "build-data-sorted.csv"
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
        "location", "component", "makeType", "bp_arch", "bp_compiler", "bp_opt",
        "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_weekofyear", "branch_prefix",
        "jobs", "localJobs", "target_cnt", "target_has_dist", "branch_id_str",
        "lag_1_grouped", "lag_max_rss_global_w5", "rolling_p95_rss_g1_w5"
    ]
    QUANT_FEATURES = [
        "build_load", "target_intensity", "debug_multiplier",
        "heavy_target_flag", "high_parallelism"
    ]
    ALL_FEATURES = list(dict.fromkeys(BASE_FEATURES + QUANT_FEATURES))

    CV_SPLITS = 3
    N_CALLS_PER_FAMILY = 1
    NUM_PARALLEL_WORKERS = 1

    MODEL_FAMILIES = {
        "qe_regression":     {"type": "regression", "base_model": "quantile_ensemble"},
        "xgboost_classification": {"type": "classification", "base_model": "xgboost"},
        "xgboost_regression":    {"type": "regression", "base_model": "xgboost"},
        "lightgbm_classification": {"type": "classification", "base_model": "lightgbm"},
        "rf_classification":     {"type": "classification", "base_model": "random_forest"},
        "lightgbm_regression":    {"type": "regression", "base_model": "lightgbm"},
        "lr_classification":   {"type": "classification", "base_model": "logistic_regression"},
    }

    # Hyperparameter configuration system
    # Each parameter can have either:
    # - "default": single value for default training
    # - "min", "max", "type": range for hyperparameter search
    # - "choices": list of categorical choices
    HYPERPARAMETER_CONFIGS = {
        # Classification-specific common parameters  
        "classification_common": {
            "n_bins": {"min": 3, "max": 15, "type": "int", "default": 5},
            "strategy": {"choices": ["uniform", "quantile", "kmeans"], "default": "quantile"},
        },
        
        # Model-specific parameters
        "quantile_ensemble": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
            "safety": {"min": 1.00, "max": 1.15, "type": "float", "default": 1.05},
            "gb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "gb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "gb_lr": {"min": 0.01, "max": 0.15, "type": "float", "log": True, "default": 0.05},
            "xgb_n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "xgb_max_depth": {"min": 3, "max": 9, "type": "int", "default": 6},
            "xgb_lr": {"min": 0.01, "max": 0.15, "type": "float", "log": True, "default": 0.05},
        },
        
        "xgboost": {
            # Regression parameters
            "regression": {
                "use_quant_feats": {"choices": [True, False], "default": True},
                "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
                "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
                "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
                "learning_rate": {"min": 0.01, "max": 0.2, "type": "float", "log": True, "default": 0.1},
            },
            # Classification parameters  
            "classification": {
                "use_quant_feats": {"choices": [True, False], "default": True},
                "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
                "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
                "learning_rate": {"min": 0.01, "max": 0.2, "type": "float", "log": True, "default": 0.1},
            }
        },
        
        "lightgbm": {
            # Regression parameters
            "regression": {
                "use_quant_feats": {"choices": [True, False], "default": True},
                "alpha": {"choices": [0.90, 0.95, 0.98, 0.99], "default": 0.95},
                "n_estimators": {"min": 100, "max": 700, "type": "int", "default": 400},
                "num_leaves": {"min": 20, "max": 60, "type": "int", "default": 31},
                "learning_rate": {"min": 0.01, "max": 0.2, "type": "float", "log": True, "default": 0.1},
            },
            # Classification parameters
            "classification": {
                "use_quant_feats": {"choices": [True, False], "default": True},
                "n_estimators": {"min": 200, "max": 800, "type": "int", "default": 400},
                "max_depth": {"min": 4, "max": 10, "type": "int", "default": 6},
                "num_leaves": {"min": 20, "max": 64, "type": "int", "default": 31},
                "learning_rate": {"min": 0.01, "max": 0.2, "type": "float", "log": True, "default": 0.1},
            }
        },
        
        "random_forest": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "n_estimators": {"min": 200, "max": 700, "type": "int", "default": 300},
            "max_depth": {"min": 6, "max": 15, "type": "int", "default": 10},
        },
        
        "logistic_regression": {
            "use_quant_feats": {"choices": [True, False], "default": True},
            "C": {"min": 1e-2, "max": 10.0, "type": "float", "log": True, "default": 1.0},
            "solver": {"choices": ["liblinear", "saga"], "default": "liblinear"},
            "penalty": {"choices": ["l1", "l2", "elasticnet"], "default": "l2"},
            "l1_ratio": {"min": 0, "max": 1, "type": "float", "default": 0.5},  # Only used with elasticnet
        }
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
        for param, config in Config.HYPERPARAMETER_CONFIGS.get(task_common_key, {}).items():
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
            return trial.suggest_int(param_name, param_config["min"], param_config["max"])
        elif param_config.get("type") == "float":
            log = param_config.get("log", False)
            return trial.suggest_float(param_name, param_config["min"], param_config["max"], log=log)
        else:
            raise ValueError(f"Invalid parameter configuration for {param_name}: {param_config}")

    @staticmethod
    def get_search_space(trial, base_model, task_type):
        """
        Defines the hyperparameter search space for a given model and task using the new configuration system.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.
            base_model (str): The base algorithm name (e.g., 'xgboost').
            task_type (str): The task type ('regression' or 'classification').

        Returns:
            dict: A dictionary of suggested hyperparameters for the trial.
        """
        params = {}
        
        # Add task-specific common parameters (currently only for classification)
        task_common_key = f"{task_type}_common"
        for param, config in Config.HYPERPARAMETER_CONFIGS.get(task_common_key, {}).items():
            params[param] = Config._suggest_param(trial, param, config)
            
        # Add model-specific parameters
        model_config = Config.HYPERPARAMETER_CONFIGS.get(base_model, {})
        if task_type in model_config:
            # Model has task-specific config
            model_params = model_config[task_type]
        else:
            # Model uses same config for all tasks
            model_params = model_config
            
        for param, config in model_params.items():
            params[param] = Config._suggest_param(trial, param, config)
            
        # Add special handling for model-specific logic
        if base_model == 'quantile_ensemble' and task_type == 'regression':
            # For quantile ensemble, alpha is already included in the model config
            pass
                
        elif base_model == 'xgboost' and task_type == 'regression':
            # For XGBoost regression, set specific objective and quantile_alpha
            params['objective'] = 'reg:quantileerror'
            if 'alpha' in params:
                params['quantile_alpha'] = params.pop('alpha')
                
        elif base_model == 'lightgbm' and task_type == 'regression':
            # For LightGBM regression, set objective to quantile
            params['objective'] = 'quantile'
            
        elif base_model == 'logistic_regression':
            # Handle special constraint for logistic regression
            if params.get('solver') == 'liblinear' and params.get('penalty') == 'elasticnet':
                raise optuna.exceptions.TrialPruned()
            if params.get('penalty') != 'elasticnet' and 'l1_ratio' in params:
                # Remove l1_ratio if penalty is not elasticnet
                params.pop('l1_ratio')
            elif params.get('penalty') == 'elasticnet' and params.get('solver') != 'saga':
                raise optuna.exceptions.TrialPruned()
                
        return params
