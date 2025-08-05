"""High-level training orchestration and evaluation helpers."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

from resource_prediction.config import Config
from resource_prediction.training.hyperparameter import OptunaOptimizer, QuantileEnsemblePredictor


class Trainer:
    """
    Orchestrates the ML pipeline: search, final evaluation, and reporting.
    """

    def __init__(self, config: Config, evaluate_all_archs: bool = False, task_type_filter: str | None = None):
        """
        Initializes the Trainer and loads all necessary data splits.

        Args:
            config (Config): The project's configuration object.
            evaluate_all_archs (bool): If True, evaluates all model architectures.
                                       Otherwise, evaluates only the two champions.
            task_type_filter (str | None): If provided, filters the pipeline to
                                           run only for 'regression' or 'classification'.
        """
        self.config = config
        self.evaluate_all_archs = evaluate_all_archs
        self.task_type_filter = task_type_filter
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()

    def _load_data(self):
        """Loads all preprocessed data splits from disk."""
        try:
            X_train = pd.read_pickle(self.config.X_TRAIN_PATH)
            y_train = pd.read_pickle(self.config.Y_TRAIN_PATH)
            X_test = pd.read_pickle(self.config.X_TEST_PATH)
            y_test = pd.read_pickle(self.config.Y_TEST_PATH)
            return X_train, y_train, X_test, y_test
        except FileNotFoundError:
            print("Error: Processed data not found. Please run preprocessing first.")
            return None, None, None, None

    def run_optimization_and_evaluation(self):
        """
        Runs the full pipeline: hyperparameter search followed by final
        evaluation on the hold-out test set and report generation.
        """
        if self.X_train is None:
            print("Exiting due to missing data.")
            return

        print("\nInitiating hyperparameter search...")
        optimizer = OptunaOptimizer(
            self.config, self.X_train, self.y_train, self.task_type_filter)
        studies = optimizer.run()

        self._evaluate_and_report(studies)

    @staticmethod
    def _allocation_metrics(allocated, true):
        """Calculates key business metrics for memory allocation."""
        under = np.sum(allocated < true)
        over = np.maximum(0, allocated - true)
        return {
            "under_pct": 100 * under / len(true) if len(true) > 0 else 0,
            "mean_gb_wasted": over.mean(),
            "total_over_pct": 100 * over.sum() / true.sum() if true.sum() > 0 else 0,
        }

    @staticmethod
    def _business_score(metrics):
        """Calculates the business score from a metrics dictionary."""
        return metrics["under_pct"] * 5 + metrics["total_over_pct"]

    @staticmethod
    def _evaluate_single_champion(study, config, X_train, y_train, X_test, y_test):
        """
        Fits and evaluates a single champion model on the hold-out set.
        This function is designed to be run in a separate process.
        """
        family_name = '_'.join(study.study_name.split('_')[:-1])
        metadata = config.MODEL_FAMILIES[family_name]
        best_params = study.best_trial.params.copy()

        use_quant = best_params.pop("use_quant_feats")
        features = config.BASE_FEATURES + \
            (config.QUANT_FEATURES if use_quant else [])
        X_train_fs, X_test_fs = X_train[features], X_test[features]
        y_train_gb, y_test_gb = y_train[config.TARGET_COLUMN_PROCESSED], y_test[config.TARGET_COLUMN_PROCESSED]

        model, alloc, fit_params = None, None, {}

        if metadata['type'] == 'regression':
            if metadata['base_model'] == 'quantile_ensemble':
                gb_params = {'n_estimators': best_params["gb_n_estimators"],
                             'max_depth': best_params["gb_max_depth"], 'learning_rate': best_params["gb_lr"], 'verbose': 0}
                xgb_params = {'n_estimators': best_params["xgb_n_estimators"],
                              'max_depth': best_params["xgb_max_depth"], 'learning_rate': best_params["xgb_lr"]}
                model = QuantileEnsemblePredictor(
                    alpha=best_params["alpha"], safety=best_params["safety"], gb_params=gb_params, xgb_params=xgb_params)
                fit_params['verbose'] = False
            else:
                if metadata['base_model'] == 'xgboost':
                    model = xgb.XGBRegressor(
                        **best_params, objective='reg:squarederror', n_jobs=-1, random_state=config.RANDOM_STATE)
                    fit_params['verbose'] = False
                elif metadata['base_model'] == 'random_forest':
                    model = RandomForestRegressor(
                        **best_params, n_jobs=-1, random_state=config.RANDOM_STATE, verbose=0)

                X_train_fs = pd.get_dummies(
                    X_train_fs, drop_first=True, dummy_na=False).astype(float)
                X_test_fs = pd.get_dummies(
                    X_test_fs, drop_first=True, dummy_na=False).astype(float)
                X_test_fs = X_test_fs.reindex(
                    columns=X_train_fs.columns, fill_value=0)

            model.fit(X_train_fs, y_train_gb, **fit_params)
            alloc = model.predict(X_test_fs)

        else:  # Classification
            if 'lr' in best_params:
                best_params['learning_rate'] = best_params.pop('lr')

            _, bin_edges = pd.qcut(
                y_train_gb, q=15, retbins=True, duplicates='drop')
            y_train_binned = pd.cut(
                y_train_gb, bins=bin_edges, labels=False, include_lowest=True, right=True)
            X_train_enc = pd.get_dummies(
                X_train_fs, drop_first=True, dummy_na=False).astype(float)
            X_test_enc = pd.get_dummies(
                X_test_fs, drop_first=True, dummy_na=False).astype(float)
            X_test_enc = X_test_enc.reindex(
                columns=X_train_enc.columns, fill_value=0)

            if metadata['base_model'] == 'xgboost':
                model = xgb.XGBClassifier(
                    **best_params, objective="multi:softmax", n_jobs=-1, random_state=config.RANDOM_STATE)
                fit_params['verbose'] = False
            elif metadata['base_model'] == 'lightgbm':
                model = lgb.LGBMClassifier(**best_params, objective="multiclass",
                                           n_jobs=-1, random_state=config.RANDOM_STATE, verbose=-1, verbosity=-1)
            elif metadata['base_model'] == 'catboost':
                model = ctb.CatBoostClassifier(**best_params, loss_function="MultiClass", thread_count=-1,
                                               random_state=config.RANDOM_STATE, verbose=0, allow_writing_files=False)
            elif metadata['base_model'] == 'random_forest':
                model = RandomForestClassifier(
                    **best_params, n_jobs=-1, random_state=config.RANDOM_STATE, verbose=0)
            elif metadata['base_model'] == 'logistic_regression':
                model = LogisticRegression(
                    **best_params, max_iter=1000, n_jobs=-1, random_state=config.RANDOM_STATE, verbose=0)

            model.fit(X_train_enc, y_train_binned, **fit_params)
            pred_class = model.predict(X_test_enc).astype(int)
            alloc = bin_edges[np.minimum(pred_class + 1, len(bin_edges) - 1)]

        hold_metrics = Trainer._allocation_metrics(alloc, y_test_gb.values)
        hold_metrics["score"] = Trainer._business_score(hold_metrics)
        result_row = {'model': family_name, 'score_cv': study.best_value, **
                      study.best_trial.params, **{f"{k}_hold": v for k, v in hold_metrics.items()}}

        return metadata['type'], result_row

    def _evaluate_and_report(self, studies):
        """
        Finds models to evaluate based on the `evaluate_all_archs` flag,
        evaluates them in parallel, and generates reports.
        """
        valid_studies = [s for s in studies if s.best_trial is not None]

        if not valid_studies:
            print("\nNo successful studies found to evaluate. Exiting.")
            return

        if self.evaluate_all_archs:
            print("\nEvaluating the best performer from EACH model architecture...")
            models_to_evaluate = valid_studies
        else:
            print("\nFinding the single best champion for each task type...")
            regression_studies = [s for s in valid_studies if self.config.MODEL_FAMILIES['_'.join(
                s.study_name.split('_')[:-1])]['type'] == 'regression']
            classification_studies = [s for s in valid_studies if self.config.MODEL_FAMILIES['_'.join(
                s.study_name.split('_')[:-1])]['type'] == 'classification']

            best_regr = min(
                regression_studies, key=lambda s: s.best_value) if regression_studies else None
            best_class = min(
                classification_studies, key=lambda s: s.best_value) if classification_studies else None
            models_to_evaluate = [s for s in [
                best_regr, best_class] if s is not None]

        print("\nThe following models will be evaluated on the hold-out set:")
        if models_to_evaluate:
            for study in models_to_evaluate:
                family_name = '_'.join(study.study_name.split('_')[:-1])
                score = study.best_value
                print(f"  - {family_name.upper()} (CV Score: {score:.4f})")
        else:
            print("  - No successful models found to evaluate.")
            return

        regression_results, classification_results = [], []

        with ProcessPoolExecutor(max_workers=len(models_to_evaluate)) as executor:
            future_to_study = {executor.submit(self._evaluate_single_champion, s, self.config,
                                               self.X_train, self.y_train, self.X_test, self.y_test): s for s in models_to_evaluate}
            progress_bar = tqdm(as_completed(future_to_study), total=len(
                models_to_evaluate), desc="Evaluating Models")

            for future in progress_bar:
                study = future_to_study[future]
                family_name = '_'.join(study.study_name.split('_')[:-1])
                progress_bar.set_postfix_str(
                    f"Completed: {family_name.upper()}")
                try:
                    task_type, result_row = future.result()
                    if task_type == 'regression':
                        regression_results.append(result_row)
                    else:
                        classification_results.append(result_row)
                except Exception as exc:
                    print(
                        f"\nModel {family_name} generated an exception: {exc}")

        print("\nFinal evaluation complete.")
        if regression_results:
            df = pd.DataFrame(regression_results)
            df.to_csv(self.config.REGRESSION_RESULTS_CSV_PATH, index=False)
            print(
                f"Regression results saved to {self.config.REGRESSION_RESULTS_CSV_PATH}")
        if classification_results:
            df = pd.DataFrame(classification_results)
            df.to_csv(self.config.CLASSIFICATION_RESULTS_CSV_PATH, index=False)
            print(
                f"Classification results saved to {self.config.CLASSIFICATION_RESULTS_CSV_PATH}")

        if self.evaluate_all_archs:
            all_results = pd.concat([pd.DataFrame(regression_results), pd.DataFrame(
                classification_results)], ignore_index=True)
            if not all_results.empty:
                plt.figure(figsize=(10, 8))
                order = all_results.sort_values("score_hold")["model"]
                sns.barplot(data=all_results, y="model",
                            x="score_hold", order=order, color="steelblue")
                plt.xlabel("Hold-out Set Business Score (Lower is Better)")
                plt.ylabel("Model Architecture")
                plt.title("Final Model Performance on Hold-out Data")
                plt.tight_layout()
                plt.savefig(self.config.RESULTS_PLOT_PATH)
                print(
                    f"Comparison chart saved to {self.config.RESULTS_PLOT_PATH}")
