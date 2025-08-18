"""Hyperparameter optimisation utilities using Optuna."""

import pandas as pd
import numpy as np
import multiprocessing
import optuna

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

from resource_prediction.config import Config
from resource_prediction.models import QuantileEnsemblePredictor


class OptunaOptimizer:
    """Orchestrates hyperparameter search for all model families using Optuna."""

    def __init__(self, config: Config, X_train: pd.DataFrame, y_train: pd.DataFrame, task_type_filter: str | None = None):
        self.config = config
        self.X_train = X_train
        self.y_train_gb = y_train[config.TARGET_COLUMN_PROCESSED]
        self.task_type_filter = task_type_filter
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

        if isinstance(model, QuantileEnsemblePredictor):
            for tr_idx, te_idx in tscv.split(X):
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                allocs.extend(model.predict(X.iloc[te_idx]))
                truths.extend(y.iloc[te_idx])
        else:
            X_encoded = pd.get_dummies(
                X, drop_first=True, dummy_na=False).astype(float)
            for tr_idx, te_idx in tscv.split(X_encoded):
                X_train_fold, X_test_fold = X_encoded.iloc[tr_idx], X_encoded.iloc[te_idx]
                y_train_fold, y_test_fold = y.iloc[tr_idx], y.iloc[te_idx]

                X_test_fold = X_test_fold.reindex(
                    columns=X_train_fold.columns, fill_value=0)

                model.fit(X_train_fold, y_train_fold)
                allocs.extend(model.predict(X_test_fold))
                truths.extend(y_test_fold)

        metrics = self._allocation_metrics(np.array(allocs), np.array(truths))
        return self._business_score(metrics)

    def _evaluate_classification(self, model, X, y, n_bins, strategy):
        """Evaluates a classification model using time-series cross-validation."""
        min_val, max_val = y.min(), y.max()
        if strategy == 'quantile':
            try:
                _, bin_edges = pd.qcut(
                    y, q=n_bins, retbins=True, duplicates='drop')
            except ValueError:
                bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        elif strategy == 'uniform':
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        else:  # kmeans
            kmeans = KMeans(n_clusters=n_bins, random_state=self.config.RANDOM_STATE, n_init='auto').fit(
                y.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            edges = [(centers[i] + centers[i+1]) /
                     2 for i in range(len(centers)-1)]
            bin_edges = np.array([min_val] + edges + [max_val])

        bin_edges = np.array(sorted(list(set(bin_edges))))
        if len(bin_edges) < 2:
            return 1e9

        y_binned = pd.cut(y, bins=bin_edges, labels=False,
                          include_lowest=True, right=True)
        X_encoded = pd.get_dummies(
            X, drop_first=True, dummy_na=False).astype(float)

        tscv = TimeSeriesSplit(n_splits=self.config.CV_SPLITS)
        allocs, truths = [], []
        for tr_idx, te_idx in tscv.split(X_encoded):
            X_train_fold, X_test_fold = X_encoded.iloc[tr_idx], X_encoded.iloc[te_idx]
            y_train_fold, y_test_fold = y_binned.iloc[tr_idx], y.iloc[te_idx]

            X_test_fold = X_test_fold.reindex(
                columns=X_train_fold.columns, fill_value=0)

            model.fit(X_train_fold, y_train_fold)
            pred_class = model.predict(X_test_fold).astype(int)
            allocs.extend(bin_edges[np.minimum(
                pred_class + 1, len(bin_edges) - 1)])
            truths.extend(y_test_fold)

        metrics = self._allocation_metrics(np.array(allocs), np.array(truths))
        return self._business_score(metrics)

    def _objective(self, trial, base_model, model_type):
        """The core objective function for Optuna to minimize."""
        params = self.config.get_search_space(trial, base_model, model_type)
        use_quant_feats = params.pop("use_quant_feats")
        X_trial = self._get_feature_set(use_quant_feats)
        model = None

        if model_type == "regression":
            if base_model == 'quantile_ensemble':
                alpha = params.pop("alpha")
                gb_params = {'n_estimators': params["gb_n_estimators"],
                             'max_depth': params["gb_max_depth"], 'learning_rate': params["gb_lr"]}
                xgb_params = {'n_estimators': params["xgb_n_estimators"],
                              'max_depth': params["xgb_max_depth"], 'learning_rate': params["xgb_lr"]}
                model = QuantileEnsemblePredictor(
                    alpha=alpha, safety=params["safety"], gb_params=gb_params, xgb_params=xgb_params, random_state=self.config.RANDOM_STATE)
            elif base_model == 'xgboost':
                model = xgb.XGBRegressor(
                    **params, n_jobs=1, random_state=self.config.RANDOM_STATE)
            elif base_model == 'lightgbm':
                model = lgb.LGBMRegressor(
                    **params, n_jobs=1, random_state=self.config.RANDOM_STATE, verbose=-1)

            if model is None:
                raise ValueError(f"Unknown regression model: {base_model}")
            return self._evaluate_regression(model, X_trial, self.y_train_gb)

        else:  # Classification
            n_bins, strategy = params.pop("n_bins"), params.pop("strategy")
            if 'lr' in params:
                params['learning_rate'] = params.pop('lr')

            if base_model == 'xgboost':
                model = xgb.XGBClassifier(
                    **params, objective="multi:softmax", n_jobs=1, random_state=self.config.RANDOM_STATE)
            elif base_model == 'lightgbm':
                model = lgb.LGBMClassifier(
                    **params, objective="multiclass", n_jobs=1, random_state=self.config.RANDOM_STATE, verbose=-1)
            elif base_model == 'random_forest':
                model = RandomForestClassifier(
                    **params, n_jobs=1, random_state=self.config.RANDOM_STATE)
            elif base_model == 'logistic_regression':
                model = LogisticRegression(
                    **params, max_iter=1000, n_jobs=1, random_state=self.config.RANDOM_STATE, multi_class="auto")

            if model is None:
                raise ValueError(f"Unknown classification model: {base_model}")
            return self._evaluate_classification(model, X_trial, self.y_train_gb, n_bins, strategy)

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
                    lambda trial: self._objective(
                        trial, metadata['base_model'], metadata['type']),
                    n_trials=remaining_trials,
                    n_jobs=self.config.NUM_PARALLEL_WORKERS,
                    show_progress_bar=True
                )
            else:
                print(
                    f"Skipping {family_name.upper()} – already optimised with {completed_trials} trials.")

            all_studies.append(study)

        return all_studies
