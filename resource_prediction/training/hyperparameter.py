"""Hyperparameter optimisation utilities using Optuna."""

import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import TimeSeriesSplit

from resource_prediction.config import Config


class OptunaOptimizer:
    """Orchestrates hyperparameter search for all model families using Optuna."""

    def __init__(self, config: Config, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                 task_type_filter: str | None = None, model_families: list[str] | None = None):
        self.config = config
        self.X_train = X_train
        self.y_train_gb = y_train[config.TARGET_COLUMN_PROCESSED]
        self.task_type_filter = task_type_filter
        self.model_families = model_families
        self.config.OPTUNA_DB_DIR.mkdir(exist_ok=True, parents=True)

    def _get_feature_set(self, use_quant_feats: bool):
        """Assembles the feature dataframe based on the trial parameter."""
        features = self.config.BASE_FEATURES + \
            (self.config.QUANT_FEATURES if use_quant_feats else [])
        return self.X_train[list(dict.fromkeys(features))]

    def _business_score(self, metrics):
        """Calculates the business score to minimize."""
        return metrics["under_pct"] * 5 + metrics["total_over_pct"]

    def _allocation_metrics(self, allocated, true):
        """Calculates key business metrics for memory allocation."""
        under = np.sum(allocated < true)
        over = np.maximum(0, allocated - true)
        return {
            "under_pct": 100 * under / len(true) if len(true) > 0 else 0,
            "total_over_pct": 100 * over.sum() / true.sum() if true.sum() > 0 else 0,
        }

    def _evaluate_regression(self, model, X, y):
        """Evaluates a regression model using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.config.CV_SPLITS)
        allocs, truths = [], []

        for tr_idx, te_idx in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[tr_idx], X.iloc[te_idx]
            y_train_fold, y_test_fold = y.iloc[tr_idx], y.iloc[te_idx]

            # All wrapper models now use the same interface
            model.fit(X_train_fold, y_train_fold)
            allocs.extend(model.predict(X_test_fold))
            truths.extend(y_test_fold)

        metrics = self._allocation_metrics(np.array(allocs), np.array(truths))
        return self._business_score(metrics)

    def _evaluate_classification(self, model, X, y):
        """Evaluates a classification model using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.config.CV_SPLITS)
        allocs, truths = [], []
        
        for tr_idx, te_idx in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[tr_idx], X.iloc[te_idx]
            y_train_fold, y_test_fold = y.iloc[tr_idx], y.iloc[te_idx]

            # Fit and predict using the wrapper model's interface
            model.fit(X_train_fold, y_train_fold)
            pred_allocs = model.predict(X_test_fold)
            
            allocs.extend(pred_allocs)
            truths.extend(y_test_fold)

        metrics = self._allocation_metrics(np.array(allocs), np.array(truths))
        return self._business_score(metrics)

    def _objective(self, trial, family_name):
        """The core objective function for Optuna to minimize."""
        params = self.config.get_search_space(trial, family_name)
        use_quant_feats = params.pop("use_quant_feats")
        X_trial = self._get_feature_set(use_quant_feats)
        
        # Get model metadata and create model dynamically - no hardcoded model creation!
        if family_name not in self.config.MODEL_FAMILIES:
            raise ValueError(f"Unknown model family: {family_name}")
            
        metadata = self.config.MODEL_FAMILIES[family_name]
        model_class = metadata['class']
        task_type = metadata['type']
        
        # Create model instance dynamically using class reference
        model = model_class(**params, random_state=self.config.RANDOM_STATE)
        
        # Evaluate based on task type
        if task_type == 'regression':
            return self._evaluate_regression(model, X_trial, self.y_train_gb)
        elif task_type == 'classification':
            return self._evaluate_classification(model, X_trial, self.y_train_gb)
        else:
            raise ValueError(f"Unknown task type '{task_type}' for model family '{family_name}'")

    def run(self):
        """
        Runs the hyperparameter search for all relevant model families. This method
        handles study resumption by first trying to load existing timestamped
        studies, then falls back to creating/loading studies with a clean,
        deterministic name.
        """
        all_studies = []
        for family_name, metadata in self.config.MODEL_FAMILIES.items():
            if self.task_type_filter and metadata['type'] != self.task_type_filter:
                continue
            if self.model_families and family_name not in self.model_families:
                continue

            storage_url = f"sqlite:///{self.config.OPTUNA_DB_DIR}/{family_name}.db"
            db_path = self.config.OPTUNA_DB_DIR / f"{family_name}.db"
            study = None

            if db_path.exists():
                try:
                    all_summaries = optuna.study.get_all_study_summaries(
                        storage=storage_url)
                    timestamped_summaries = [
                        s for s in all_summaries if s.study_name.startswith(f"{family_name}_")]

                    if timestamped_summaries:
                        best_timestamped_study = max(
                            timestamped_summaries, key=lambda s: s.n_trials)
                        print(
                            f"Resuming existing timestamped study '{best_timestamped_study.study_name}' for model family '{family_name}'.")
                        study = optuna.load_study(
                            study_name=best_timestamped_study.study_name,
                            storage=storage_url
                        )
                except Exception as e:
                    print(
                        f"Warning: Could not read existing database at {db_path}. Error: {e}")

            if study is None:
                study_name = family_name  # The clean, deterministic name
                print(
                    f"Loading or creating new study '{study_name}' for model family '{family_name}'.")
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(
                        seed=self.config.RANDOM_STATE),
                    load_if_exists=True,
                )

            completed_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining_trials = self.config.N_CALLS_PER_FAMILY - completed_trials

            if remaining_trials > 0:
                print(
                    f"Optimising {family_name.upper()} – running {remaining_trials} more trials...")
                study.optimize(
                    lambda trial: self._objective(trial, family_name),
                    n_trials=remaining_trials,
                    n_jobs=self.config.NUM_PARALLEL_WORKERS,
                    show_progress_bar=True
                )
            else:
                print(
                    f"Skipping {family_name.upper()} – already optimised with {completed_trials} trials.")

            all_studies.append(study)

        return all_studies
