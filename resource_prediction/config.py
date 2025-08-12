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
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "resource_prediction" / "models"
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
    N_CALLS_PER_FAMILY = 120
    NUM_PARALLEL_WORKERS = 32

    MODEL_FAMILIES = {
        "qe_regression":     {"type": "regression", "base_model": "quantile_ensemble"},
        "xgboost_classification": {"type": "classification", "base_model": "xgboost"},
        "xgboost_regression":    {"type": "regression", "base_model": "xgboost"},
        "lightgbm_classification": {"type": "classification", "base_model": "lightgbm"},
        "rf_classification":     {"type": "classification", "base_model": "random_forest"},
        "rf_regression":         {"type": "regression", "base_model": "random_forest"},
        "lr_classification":   {"type": "classification", "base_model": "logistic_regression"},
    }

    @staticmethod
    def get_search_space(trial, base_model, task_type):
        """
        Defines the hyperparameter search space for a given model and task.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.
            base_model (str): The base algorithm name (e.g., 'xgboost').
            task_type (str): The task type ('regression' or 'classification').

        Returns:
            dict: A dictionary of suggested hyperparameters for the trial.
        """
        use_quant = trial.suggest_categorical("use_quant_feats", [True, False])

        if task_type == 'regression':
            if base_model == 'quantile_ensemble':
                return {
                    "alpha": trial.suggest_categorical("alpha", [0.90, 0.95, 0.98, 0.99]),
                    "safety": trial.suggest_float("safety", 1.00, 1.15),
                    "gb_n_estimators": trial.suggest_int("gb_n_estimators", 200, 700),
                    "gb_max_depth": trial.suggest_int("gb_max_depth", 3, 9),
                    "gb_lr": trial.suggest_float("gb_lr", 0.01, 0.15, log=True),
                    "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 200, 700),
                    "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 9),
                    "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
                    "use_quant_feats": use_quant,
                }
            if base_model == 'xgboost':
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "alpha": trial.suggest_float("alpha", 0.5, 1.0),
                    "use_quant_feats": use_quant,
                }
            if base_model == 'random_forest':
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "use_quant_feats": use_quant,
                }

        elif task_type == 'classification':
            common_params = {
                "n_bins": trial.suggest_int("n_bins", 3, 15),
                "strategy": trial.suggest_categorical("strategy", ["uniform", "quantile", "kmeans"]),
                "use_quant_feats": use_quant
            }

            if base_model in ['xgboost', 'lightgbm']:
                lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
                if base_model == 'xgboost':
                    model_params = {"n_estimators": trial.suggest_int(
                        "n_estimators", 200, 800), "max_depth": trial.suggest_int("max_depth", 4, 10), "lr": lr}
                else:  # lightgbm
                    model_params = {"n_estimators": trial.suggest_int("n_estimators", 200, 800), "max_depth": trial.suggest_int(
                        "max_depth", 4, 10), "num_leaves": trial.suggest_int("num_leaves", 20, 64), "lr": lr}
                return {**common_params, **model_params}

            if base_model == 'random_forest':
                model_params = {"n_estimators": trial.suggest_int(
                    "n_estimators", 200, 700), "max_depth": trial.suggest_int("max_depth", 6, 15)}
                return {**common_params, **model_params}

            if base_model == 'logistic_regression':
                solver = trial.suggest_categorical(
                    "solver", ["liblinear", "saga"])

                penalty = trial.suggest_categorical(
                    "penalty", ["l1", "l2", "elasticnet"])

                if solver == "liblinear" and penalty == "elasticnet":
                    raise optuna.exceptions.TrialPruned()

                model_params = {
                    "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
                    "solver": solver,
                    "penalty": penalty,
                }

                if penalty == "elasticnet":
                    if solver != "saga":
                        raise optuna.exceptions.TrialPruned()
                    model_params["l1_ratio"] = trial.suggest_float(
                        "l1_ratio", 0, 1)

                return {**common_params, **model_params}

        return {}
