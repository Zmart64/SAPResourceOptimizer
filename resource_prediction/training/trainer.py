"""High-level training orchestration and evaluation helpers."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import joblib

from resource_prediction.config import Config
from resource_prediction.training.hyperparameter import OptunaOptimizer
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

    def __init__(self, config: Config, evaluate_all_archs: bool = False, task_type_filter: str | None = None,
                 save_models: bool = False, model_families: list[str] | None = None, use_defaults: bool = False):
        """
        Initializes the Trainer and loads data splits and baseline statistics.

        Args:
            config: Configuration object
            evaluate_all_archs: Whether to evaluate all architectures
            task_type_filter: Filter by task type ('regression' or 'classification')
            save_models: Whether to save trained models
            model_families: List of specific model families to run (e.g., ['xgboost_regression', 'rf_classification'])
            use_defaults: If True, train models with default hyperparameters instead of running hyperparameter search
        """
        self.config = config
        self.evaluate_all_archs = evaluate_all_archs
        self.task_type_filter = task_type_filter
        self.save_models = save_models
        self.model_families = model_families
        self.use_defaults = use_defaults
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
        Runs hyperparameter search or default training, evaluates all found architectures, and updates
        the persistent champion result files. This is the only function that modifies
        the champion CSV files.
        """
        if self.X_train is None:
            print("Exiting due to missing data.")
            return

        if self.use_defaults:
            print("\nTraining models with default hyperparameters...")
            studies = self._train_with_defaults()
        else:
            print("\nInitiating hyperparameter search...")
            optimizer = OptunaOptimizer(
                self.config, self.X_train, self.y_train, self.task_type_filter, self.model_families)
            studies = optimizer.run()

        print("\nSearch/Training complete. Evaluating all architectures to update champion files...")
        regr_results_df, class_results_df, all_model_stats = self._evaluate_and_report(
            studies, force_evaluate_all=True)

        if not regr_results_df.empty:
            self._update_and_save_champion_results(
                regr_results_df, self.config.REGRESSION_RESULTS_CSV_PATH)
        if not class_results_df.empty:
            self._update_and_save_champion_results(
                class_results_df, self.config.CLASSIFICATION_RESULTS_CSV_PATH)

        # After saving champion CSVs, generate the unified summary report which merges
        # CV/holdout/timing metrics from those files.
        if self.baseline_stats and all_model_stats:
            print("\nGenerating unified allocation reports for this run...")
            report_data = [self.baseline_stats] + all_model_stats
            generate_summary_report(
                report_data, self.config.ALLOCATION_SUMMARY_REPORT_PATH)

    def _train_with_defaults(self):
        """
        Train models using default hyperparameters instead of running hyperparameter search.
        Returns mock studies that can be used with the existing evaluation pipeline.
        """
        studies = []

        # Get the model families to train
        families_to_train = self.config.MODEL_FAMILIES.items()
        if self.model_families:
            families_to_train = [(name, metadata) for name, metadata in families_to_train
                                 if name in self.model_families]

        for family_name, metadata in families_to_train:
            if self.task_type_filter and metadata['type'] != self.task_type_filter:
                continue

            print(f"Training {family_name} with default parameters...")

            # Get default parameters
            default_params = self.config.get_defaults(family_name)
            # Use same objective pathway as hyperparameter search for consistency
            optimizer = OptunaOptimizer(self.config, self.X_train, self.y_train)

            class DefaultTrial:
                def __init__(self, params):
                    self.params = params
                def suggest_categorical(self, name, choices):
                    return self.params.get(name, choices[0])
                def suggest_int(self, name, low, high):
                    return self.params.get(name, (low + high) // 2)
                def suggest_float(self, name, low, high, log=False, step=None):
                    return self.params.get(name, (low + high) / 2)

            trial = DefaultTrial(default_params)
            try:
                score = optimizer._objective(trial, family_name)
                mock_trial = MockTrial(trial.params.copy(), score)
                mock_study = MockStudy(family_name, mock_trial)
                studies.append(mock_study)

                print(f"  {family_name}: CV Score = {score:.4f}")

            except Exception as e:
                print(f"  Error training {family_name}: {e}")
                continue

        return studies

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

            # Use clean configuration system: only get parameters this model family needs
            try:
                # Start with default parameters for this family
                clean_params = self.config.get_defaults(family_name)

                # Override defaults with values from CSV where available
                family_config = self.config.HYPERPARAMETER_CONFIGS.get(
                    family_name, {})
                for param_name in family_config.keys():
                    if param_name in row and not pd.isna(row[param_name]):
                        value = row[param_name]
                        # Type conversion
                        if isinstance(value, float) and value.is_integer():
                            value = int(value)
                        elif str(value).lower() in ['true', 'false']:
                            value = str(value).lower() == 'true'
                        clean_params[param_name] = value

                mock_studies.append(
                    MockStudy(family_name, MockTrial(clean_params, cv_score)))

            except Exception as e:
                print(
                    f"Warning: Could not load parameters for {family_name}: {e}")
                continue

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

        # Get feature selection
        use_quant = best_params.pop("use_quant_feats", True)
        base_features = config.BASE_FEATURES + \
            (config.QUANT_FEATURES if use_quant else [])
        X_train_fs, X_test_fs = X_train[base_features], X_test[base_features]
        y_train_gb, y_test_gb = y_train[config.TARGET_COLUMN_PROCESSED], y_test[config.TARGET_COLUMN_PROCESSED]

        # Create the model dynamically using the same system as hyperparameter search
        model_class = metadata['class']

        confidence_threshold = None
        if task_type == 'classification':
            # For classification, threshold is mandatory
            confidence_threshold = best_params.pop('confidence_threshold')
        else:
            # For regression, it might be present from a mixed run, so pop it safely
            best_params.pop('confidence_threshold', None)

        model = model_class(**best_params, random_state=config.RANDOM_STATE)

        # Set the threshold on the model instance if it's a classification model
        if task_type == 'classification':
            model.confidence_threshold = confidence_threshold

        # Fit the model
        model.fit(X_train_fs, y_train_gb)
        # Time predictions to compute average prediction time per sample
        start_time = time.time()
        if task_type == 'classification' and confidence_threshold is not None:
            alloc = model.predict(X_test_fs, confidence_threshold=confidence_threshold)
        else:
            alloc = model.predict(X_test_fs)
        end_time = time.time()
        avg_pred_time = (end_time - start_time) / len(alloc) if len(alloc) > 0 else 0

        # Handle saving the model
        if save_model:
            os.makedirs(config.MODELS_DIR, exist_ok=True)

            # For models that need feature encoding, get the encoded features
            if hasattr(model, 'columns') and model.columns is not None:
                features = model.columns
            else:
                features = base_features

            # Create and fit preprocessing pipeline
            preprocessor = ModelPreprocessor(
                categorical_features=['location', 'component',
                                      'makeType', 'bp_arch', 'bp_compiler', 'bp_opt'],
                expected_features=features
            )
            # Fit preprocessor on encoded data for consistency
            if hasattr(model, '_encode'):
                X_train_encoded = model._encode(X_train_fs, fit=False)
                preprocessor.fit(X_train_encoded)
            else:
                preprocessor.fit(X_train_fs)

            # Create deployable model wrapper
            deployable_model = DeployableModel(
                model=model,
                model_type=base_model_name,
                task_type=task_type,
                preprocessor=preprocessor,
                bin_edges=getattr(model, 'bin_edges', None),
                metadata={
                    'training_features': features,
                    'model_family': family_name,
                    'training_timestamp': str(pd.Timestamp.now())
                }
            )

            # Save deployable model
            deployable_model.save(config.MODELS_DIR / f"{family_name}.pkl")

            print(f"Saved deployable model artifact for '{family_name}'")

        # Calculate metrics
        model_alloc_stats = calculate_allocation_categories(
            name=family_name, allocations=alloc, true_values=y_test_gb.values)
        hold_metrics = Trainer._allocation_metrics(alloc, y_test_gb.values)
        hold_metrics["score"] = Trainer._business_score(hold_metrics)

        # Ensure the confidence_threshold is included in the results for classification
        final_params = study.best_trial.params.copy()
        if task_type == 'classification':
            final_params['confidence_threshold'] = confidence_threshold

        # Construct result row including average prediction time
        result_row = {
            'model': family_name,
            'score_cv': study.best_value,
            'avg_pred_time': avg_pred_time,
            **final_params,
            **{f"{k}_hold": v for k, v in hold_metrics.items()}
        }

        return task_type, result_row, model_alloc_stats

    def _evaluate_and_report(self, studies, force_evaluate_all: bool = False):
        """
        Performs the core evaluation logic and returns the results as DataFrames.
        It does NOT save the champion result CSVs itself.
        """
        def _has_valid_best_trial(study):
            """Safely check if a study has a valid best trial."""
            try:
                return hasattr(study, 'best_trial') and study.best_trial is not None
            except (ValueError, AttributeError):
                # ValueError: raised when no trials are complete
                # AttributeError: raised when best_trial doesn't exist
                return False

        valid_studies = [s for s in studies if _has_valid_best_trial(s)]
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
            # Generate allocation comparison plot now; summary report will be generated
            # after champion CSVs are saved so it can include merged metrics.
            print("\nGenerating allocation comparison plot for this run...")
            report_data = [self.baseline_stats] + all_model_stats
            plot_path = self.config.ALLOCATION_PLOT_PATH
            plot_allocation_comparison(report_data, plot_path)

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
                print(f"Comparison chart saved to {self.config.RESULTS_PLOT_PATH}")
                # Scatter plot: Hold-out score vs prediction time
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=all_results, x='avg_pred_time', y='score_hold', hue='model', s=100)
                plt.xlabel('Average Prediction Time (s)', fontsize=12)
                plt.ylabel('Hold-out Set Business Score (Lower is Better)', fontsize=12)
                plt.title('Prediction Time vs Model Performance', fontsize=14)
                plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(self.config.SCORE_TIME_PLOT_PATH)
                print(f"Score vs prediction time plot saved to {self.config.SCORE_TIME_PLOT_PATH}")

        return regr_df, class_df, all_model_stats
