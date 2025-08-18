"""High-level training orchestration and evaluation helpers."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

from resource_prediction.config import Config
from resource_prediction.training.hyperparameter import OptunaOptimizer, QuantileEnsemblePredictor
from resource_prediction.models import DeployableModel
from resource_prediction.preprocessing import ModelPreprocessor
from resource_prediction.reporting import plot_allocation_comparison, generate_summary_report, calculate_allocation_categories


class MockTrial:
    """A mock of optuna.trial.Trial for re-running evaluations."""

    def __init__(self, params, value):
        self.params = params
        self.value = value


class MockStudy:
    """A mock of optuna.study.Study for re-running evaluations."""

    def __init__(self, study_name, best_trial):
        self.study_name = study_name
        self.best_trial = best_trial

    @property
    def best_value(self):
        """Mimics the best_value property of a real Optuna study."""
        return self.best_trial.value


class Trainer:
    """
    Orchestrates the ML pipeline: search, final evaluation, and reporting.
    """

    def __init__(self, config: Config, evaluate_all_archs: bool = False, task_type_filter: str | None = None, save_models: bool = False):
        """
        Initializes the Trainer and loads data splits and baseline statistics.
        """
        self.config = config
        self.evaluate_all_archs = evaluate_all_archs
        self.task_type_filter = task_type_filter
        self.save_models = save_models
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()

        self.baseline_stats = None
        if self.config.BASELINE_STATS_PATH.exists():
            self.baseline_stats = joblib.load(self.config.BASELINE_STATS_PATH)
            print(
                "Successfully loaded baseline allocation statistics. Reporting is ENABLED.")
        else:
            print(
                "Warning: Baseline allocation statistics not found. Reporting will be SKIPPED.")

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

    def _get_family_name_from_study(self, study) -> str:
        """
        Robustly extracts the base family name from a study object, handling both
        clean names ('xgboost_regression') and timestamped names
        ('xgboost_regression_20250812').
        """
        for family_key in self.config.MODEL_FAMILIES:
            if study.study_name.startswith(family_key):
                return family_key
        raise ValueError(
            f"Could not determine family name for study '{study.study_name}'")

    def run_optimization_and_evaluation(self):
        """
        Runs hyperparameter search, evaluates all found architectures, and updates
        the persistent champion result files. This is the only function that modifies
        the champion CSV files.
        """
        if self.X_train is None:
            print("Exiting due to missing data.")
            return

        print("\nInitiating hyperparameter search...")
        optimizer = OptunaOptimizer(
            self.config, self.X_train, self.y_train, self.task_type_filter)
        studies = optimizer.run()

        print("\nSearch complete. Evaluating all architectures to update champion files...")
        regr_results_df, class_results_df, _ = self._evaluate_and_report(
            studies, force_evaluate_all=True)

        if not regr_results_df.empty:
            self._update_and_save_champion_results(
                regr_results_df, self.config.REGRESSION_RESULTS_CSV_PATH)
        if not class_results_df.empty:
            self._update_and_save_champion_results(
                class_results_df, self.config.CLASSIFICATION_RESULTS_CSV_PATH)

    def run_evaluation_from_files(self):
        """
        Runs evaluation by READING from the persistent champion result files,
        without modifying them. It generates reports for the specified evaluation run.
        """
        print("\nInitiating evaluation using parameters from champion result files...")
        if self.X_train is None:
            print("Exiting due to missing data.")
            return

        all_results_df = pd.DataFrame()
        for file_path in [self.config.REGRESSION_RESULTS_CSV_PATH, self.config.CLASSIFICATION_RESULTS_CSV_PATH]:
            if file_path.exists():
                all_results_df = pd.concat(
                    [all_results_df, pd.read_csv(file_path)], ignore_index=True)

        if all_results_df.empty:
            print(
                "\nError: No champion result files found. Run a hyperparameter search first with --run-search.")
            return

        print(
            f"\nFound {len(all_results_df)} model architectures in champion files.")

        mock_studies = []
        for _, row in all_results_df.iterrows():
            family_name = row['model']
            cv_score = row['score_cv']
            param_cols = [c for c in all_results_df.columns if c not in [
                'model', 'score_cv'] and not c.endswith('_hold')]
            params = row[param_cols].to_dict()
            for key, value in params.items():
                if pd.isna(value):
                    continue
                if isinstance(value, float) and value.is_integer():
                    params[key] = int(value)
                if str(value).lower() in ['true', 'false']:
                    params[key] = str(value).lower() == 'true'
            mock_studies.append(
                MockStudy(family_name, MockTrial(params, cv_score)))

        self._evaluate_and_report(
            mock_studies, force_evaluate_all=self.evaluate_all_archs)

    def _update_and_save_champion_results(self, new_results_df: pd.DataFrame, path: Path):
        """
        Reads an existing result file, merges new results, keeps only the best
        entry for each model (based on CV score), and saves the file.
        """
        if path.exists():
            old_results_df = pd.read_csv(path)
            combined_df = pd.concat(
                [old_results_df, new_results_df], ignore_index=True)
        else:
            combined_df = new_results_df

        best_indices = combined_df.groupby('model')['score_cv'].idxmin()
        final_df = combined_df.loc[best_indices].sort_values(
            'model').reset_index(drop=True)

        final_df.to_csv(path, index=False)
        print(f"Champion results file updated and saved to {path}")

    @staticmethod
    def _allocation_metrics(allocated, true):
        """Calculates key business metrics for memory allocation."""
        under = np.sum(allocated < true)
        over = np.maximum(0, allocated - true)
        return {"under_pct": 100 * under / len(true) if len(true) > 0 else 0, "mean_gb_wasted": over.mean(), "total_over_pct": 100 * over.sum() / true.sum() if true.sum() > 0 else 0}

    @staticmethod
    def _business_score(metrics):
        """Calculates the business score from a metrics dictionary."""
        return metrics["under_pct"] * 5 + metrics["total_over_pct"]

    def _evaluate_single_champion(self, study, config, X_train, y_train, X_test, y_test, save_model: bool):
        """
        Fits, evaluates, and computes allocation stats for a single model.
        This function is designed to be run in a separate process.
        """
        family_name = self._get_family_name_from_study(study)
        metadata = config.MODEL_FAMILIES[family_name]
        task_type, base_model_name = metadata['type'], metadata['base_model']
        best_params = study.best_trial.params.copy()

        use_quant = best_params.pop("use_quant_feats")
        base_features = config.BASE_FEATURES + \
            (config.QUANT_FEATURES if use_quant else [])
        X_train_fs, X_test_fs = X_train[base_features], X_test[base_features]
        y_train_gb, y_test_gb = y_train[config.TARGET_COLUMN_PROCESSED], y_test[config.TARGET_COLUMN_PROCESSED]

        if base_model_name != 'quantile_ensemble':
            X_train_fs = pd.get_dummies(
                X_train_fs, drop_first=True, dummy_na=False).astype(float)
            X_test_fs = pd.get_dummies(
                X_test_fs, drop_first=True, dummy_na=False).astype(float)
            X_test_fs = X_test_fs.reindex(
                columns=X_train_fs.columns, fill_value=0)
            features = X_train_fs.columns.tolist()
        else:
            features = base_features

        model, bin_edges, alloc = None, None, None

        if task_type == 'regression':
            if base_model_name == 'quantile_ensemble':
                alpha = best_params.pop("alpha")
                gb_params = {'n_estimators': best_params["gb_n_estimators"],
                             'max_depth': best_params["gb_max_depth"], 'learning_rate': best_params["gb_lr"], 'verbose': 0}
                xgb_params = {'n_estimators': best_params["xgb_n_estimators"],
                              'max_depth': best_params["xgb_max_depth"], 'learning_rate': best_params["xgb_lr"]}
                model = QuantileEnsemblePredictor(
                    alpha=alpha, safety=best_params["safety"], gb_params=gb_params, xgb_params=xgb_params)
            elif base_model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    **best_params, random_state=config.RANDOM_STATE)
            elif base_model_name == 'lightgbm':
                model = lgb.LGBMRegressor(
                    **best_params, random_state=config.RANDOM_STATE, verbose=-1)

            if base_model_name not in ['random_forest']:
                model.fit(X_train_fs, y_train_gb)
                alloc = model.predict(X_test_fs)
        elif task_type == 'classification':
            n_bins, strategy = best_params.pop(
                "n_bins"), best_params.pop("strategy")
            if 'lr' in best_params:
                best_params['learning_rate'] = best_params.pop('lr')
            min_val, max_val = y_train_gb.min(), y_train_gb.max()
            if strategy == 'quantile':
                try:
                    _, bin_edges = pd.qcut(
                        y_train_gb, q=n_bins, retbins=True, duplicates='drop')
                except ValueError:
                    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            elif strategy == 'uniform':
                bin_edges = np.linspace(min_val, max_val, n_bins + 1)
            else:  # kmeans
                kmeans = KMeans(n_clusters=n_bins, random_state=config.RANDOM_STATE, n_init='auto').fit(
                    y_train_gb.values.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                edges = [(centers[i] + centers[i+1]) /
                         2 for i in range(len(centers)-1)]
                bin_edges = np.array([min_val] + edges + [max_val])

            bin_edges = np.array(sorted(list(set(bin_edges))))
            y_train_binned = pd.cut(
                y_train_gb, bins=bin_edges, labels=False, include_lowest=True, right=True)

            if base_model_name == 'lightgbm':
                model = lgb.LGBMClassifier(
                    **best_params, random_state=config.RANDOM_STATE, verbose=-1)
            else:
                model_class = {'xgboost': xgb.XGBClassifier, 'random_forest': RandomForestClassifier,
                               'logistic_regression': LogisticRegression}[base_model_name]
                model = model_class(
                    **best_params, random_state=config.RANDOM_STATE)

            model.fit(X_train_fs, y_train_binned)
            pred_class = model.predict(X_test_fs).astype(int)
            alloc = bin_edges[np.minimum(pred_class + 1, len(bin_edges) - 1)]

        if save_model:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            
            # Create and fit preprocessing pipeline
            preprocessor = ModelPreprocessor(
                categorical_features=['location', 'component', 'makeType', 'bp_arch', 'bp_compiler', 'bp_opt'],
                expected_features=features
            )
            preprocessor.fit(X_train_fs)
            
            # Create deployable model wrapper
            deployable_model = DeployableModel(
                model=model,
                model_type=base_model_name,
                task_type=task_type,
                preprocessor=preprocessor,
                bin_edges=bin_edges,
                metadata={
                    'training_features': features,
                    'model_family': family_name,
                    'training_timestamp': str(pd.Timestamp.now())
                }
            )
            
            # Save deployable model
            deployable_model.save(config.MODELS_DIR / f"{family_name}.pkl")
            
            print(f"Saved deployable model artifact for '{family_name}'")

        model_alloc_stats = calculate_allocation_categories(
            name=family_name, allocations=alloc, true_values=y_test_gb.values)
        hold_metrics = Trainer._allocation_metrics(alloc, y_test_gb.values)
        hold_metrics["score"] = Trainer._business_score(hold_metrics)
        result_row = {'model': family_name, 'score_cv': study.best_value, **
                      study.best_trial.params, **{f"{k}_hold": v for k, v in hold_metrics.items()}}

        return task_type, result_row, model_alloc_stats

    def _evaluate_and_report(self, studies, force_evaluate_all: bool = False):
        """
        Performs the core evaluation logic and returns the results as DataFrames.
        It does NOT save the champion result CSVs itself.
        """
        valid_studies = [s for s in studies if hasattr(
            s, 'best_trial') and s.best_trial]
        if not valid_studies:
            print("\nNo successful models found to evaluate.")
            return pd.DataFrame(), pd.DataFrame(), []

        regr_studies = [s for s in valid_studies if self.config.MODEL_FAMILIES[self._get_family_name_from_study(
            s)]['type'] == 'regression']
        class_studies = [s for s in valid_studies if self.config.MODEL_FAMILIES[self._get_family_name_from_study(
            s)]['type'] == 'classification']

        models_to_evaluate = []
        should_eval_all = force_evaluate_all or self.evaluate_all_archs
        if should_eval_all:
            print(
                "\nEvaluating the best performer from EACH available model architecture...")
            if self.task_type_filter != 'classification':
                models_to_evaluate.extend(regr_studies)
            if self.task_type_filter != 'regression':
                models_to_evaluate.extend(class_studies)
        else:
            print("\nFinding the single best champion for each available task type...")
            if self.task_type_filter != 'classification' and regr_studies:
                models_to_evaluate.append(
                    min(regr_studies, key=lambda s: s.best_value))
            if self.task_type_filter != 'regression' and class_studies:
                models_to_evaluate.append(
                    min(class_studies, key=lambda s: s.best_value))

        if not models_to_evaluate:
            print("\nNo models selected for evaluation based on the current filters.")
            return pd.DataFrame(), pd.DataFrame(), []

        print("\nThe following models will be evaluated on the hold-out set:")
        for study in models_to_evaluate:
            family_name = self._get_family_name_from_study(study)
            print(
                f"  - {family_name.upper()} (CV Score: {study.best_value:.4f})")

        print("\nSubmitting model evaluation tasks to be run in parallel...")
        regr_results, class_results, all_model_stats = [], [], []

        max_workers = len(models_to_evaluate) if models_to_evaluate else 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_study = {executor.submit(self._evaluate_single_champion, s, self.config, self.X_train,
                                               self.y_train, self.X_test, self.y_test, self.save_models): s for s in models_to_evaluate}
            progress_bar = tqdm(as_completed(future_to_study), total=len(
                models_to_evaluate), desc="Evaluating Models")
            for future in progress_bar:
                try:
                    task_type, result_row, model_stats = future.result()
                    if task_type == 'regression':
                        regr_results.append(result_row)
                    else:
                        class_results.append(result_row)
                    all_model_stats.append(model_stats)
                except Exception as exc:
                    family_name = future_to_study[future].study_name
                    print(
                        f"\nModel {family_name} generated an exception: {exc}")

        print("\nFinal evaluation complete.")
        regr_df = pd.DataFrame(regr_results)
        class_df = pd.DataFrame(class_results)

        if self.baseline_stats and all_model_stats:
            print("\nGenerating unified allocation reports for this run...")
            report_data = [self.baseline_stats] + all_model_stats
            plot_path = self.config.ALLOCATION_PLOT_PATH
            plot_allocation_comparison(report_data, plot_path)
            generate_summary_report(
                report_data, self.config.ALLOCATION_SUMMARY_REPORT_PATH)

        if should_eval_all and not regr_df.empty and not class_df.empty:
            all_results = pd.concat([regr_df, class_df], ignore_index=True)
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

        return regr_df, class_df, all_model_stats
