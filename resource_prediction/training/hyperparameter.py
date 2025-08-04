import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
import xgboost
import lightgbm
import catboost

from skopt import gp_minimize
from skopt.utils import use_named_args

from resource_prediction.config import Config


class BayesianOptimizer:
    """
    Manages the Bayesian hyperparameter optimization process.
    """

    def __init__(self, config: Config, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.config = config
        self.X = X_train
        self.y_gb = y_train[self.config.TARGET_COLUMN_PROCESSED]
        self.all_best_results = []
        self.pbar_inner = None

    def _evaluate_single_run(self, bin_edges, model_name, confidence_threshold, model_hyperparams):
        """Trains and evaluates a model for a single set of hyperparameters using CV."""
        if len(bin_edges) < 2:
            return None
        num_classes = len(bin_edges) - 1

        y_binned = pd.cut(self.y_gb, bins=bin_edges, labels=range(num_classes),
                          right=False, include_lowest=True, duplicates='drop')
        valid_indices = y_binned.dropna().index
        if len(valid_indices) < self.config.N_SPLITS_CV * 10:
            return None

        X_current = self.X.loc[valid_indices].reset_index(drop=True)
        y_current = y_binned.loc[valid_indices].reset_index(
            drop=True).astype(int)

        if model_name == 'xgboost':
            model = xgboost.XGBClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, n_jobs=-1)
        elif model_name == 'lightgbm':
            model = lightgbm.LGBMClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            model = catboost.CatBoostClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, verbose=0, thread_count=-1)

        tscv = TimeSeriesSplit(n_splits=self.config.N_SPLITS_CV)
        all_allocated_mem, all_true_mem = [], []

        for train_index, test_index in tscv.split(X_current, y_current):
            X_train, X_test = X_current.iloc[train_index], X_current.iloc[test_index]
            y_train, _ = y_current.iloc[train_index], y_current.iloc[test_index]
            try:
                model.fit(X_train, y_train)
                y_pred_probs = model.predict_proba(X_test)
                y_pred_base = np.argmax(y_pred_probs, axis=1)
                y_pred_final = [min(pred + 1, num_classes - 1) if conf < confidence_threshold else pred
                                for pred, conf in zip(y_pred_base, y_pred_probs[np.arange(len(y_pred_base)), y_pred_base])]

                allocated_mem_pred = np.array(
                    [bin_edges[min(c + 1, num_classes)] for c in y_pred_final])
                true_mem_test = self.y_gb.iloc[test_index].values
                all_allocated_mem.extend(allocated_mem_pred)
                all_true_mem.extend(true_mem_test)
            except Exception:
                continue

        if not all_true_mem:
            return None

        allocated_arr, true_arr = np.array(
            all_allocated_mem), np.array(all_true_mem)
        jobs_under = np.sum(allocated_arr < true_arr)
        over_allocation_gb = np.sum(np.maximum(0, allocated_arr - true_arr))
        total_true_used_gb = np.sum(true_arr)

        return {
            'perc_under_allocation': (jobs_under / len(true_arr)) * 100,
            'perc_total_over_allocation': (over_allocation_gb / total_true_used_gb) * 100 if total_true_used_gb > 0 else 0,
        }

    def _objective(self, model_name, **params):
        """The objective function for skopt to minimize."""
        self.pbar_inner.set_description(
            f"Eval: {params['strategy']}, {params['n_bins']} bins, thr={params['confidence_threshold']:.2f}")
        model_hyperparams = {k: v for k, v in params.items(
        ) if k not in ['n_bins', 'strategy', 'confidence_threshold']}
        min_val, max_val = self.y_gb.min(), self.y_gb.max()

        if params['strategy'] == 'quantile':
            _, bin_edges = pd.qcut(
                self.y_gb, q=params['n_bins'], retbins=True, duplicates='drop')
        elif params['strategy'] == 'uniform':
            bin_edges = np.linspace(min_val, max_val, params['n_bins'] + 1)
        else:  # kmeans
            kmeans = KMeans(n_clusters=params['n_bins'], random_state=self.config.RANDOM_STATE, n_init='auto').fit(
                self.y_gb.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            edges = [(centers[i] + centers[i+1]) /
                     2 for i in range(len(centers)-1)]
            bin_edges = np.array([min_val] + edges + [max_val])
        bin_edges = sorted(list(set(bin_edges)))

        results = self._evaluate_single_run(
            bin_edges, model_name, params['confidence_threshold'], model_hyperparams)
        if results is None:
            return 1e6

        under_alloc_penalty = 20.0
        score = (results['perc_under_allocation'] *
                 under_alloc_penalty) + results['perc_total_over_allocation']
        return score

    def _save_summary(self):
        """Analyzes and saves the best result to a CSV."""
        print("\n--- Optimization Complete - Determining Best Model ---")
        if not self.all_best_results:
            print("No models were successfully optimized.")
            return

        overall_best = sorted(self.all_best_results,
                              key=lambda x: x['score'])[0]
        print(
            f"\n--> Overall Best Model: {overall_best['model_name'].upper()}")
        print(f"--> Best Score (lower is better): {overall_best['score']:.4f}")

        self.config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        summary_df = pd.DataFrame([{
            'model_name': overall_best['model_name'],
            'best_params_from_grid': str(overall_best['params']),
            'optimization_score': overall_best['score']
        }])
        summary_df.to_csv(self.config.OPTIMIZATION_SUMMARY_PATH, index=False)
        print(
            f"\nOptimization summary saved to: {self.config.OPTIMIZATION_SUMMARY_PATH.resolve()}")

    def run(self):
        """Main method to run the entire optimization process for all configured models."""
        for model_name in self.config.MODELS_TO_OPTIMIZE:
            print(f"\n{'='*20} Optimizing for: {model_name.upper()} {'='*20}")
            space = self.config.SEARCH_SPACES[model_name]
            objective_for_model = partial(
                self._objective, model_name=model_name)
            self.pbar_inner = tqdm(
                total=self.config.N_OPTIMIZATION_CALLS_PER_MODEL, desc=f"Optimizing {model_name}")

            def pbar_callback(res):
                self.pbar_inner.update(1)
                self.pbar_inner.set_postfix_str(f"Best Score: {res.fun:.4f}")

            result = gp_minimize(func=use_named_args(space)(objective_for_model), dimensions=space,
                                 n_calls=self.config.N_OPTIMIZATION_CALLS_PER_MODEL,
                                 random_state=self.config.RANDOM_STATE, n_jobs=-1, callback=[pbar_callback])
            self.pbar_inner.close()

            best_params = {space[i].name: result.x[i]
                           for i in range(len(space))}
            self.all_best_results.append(
                {'model_name': model_name, 'score': result.fun, 'params': best_params})

        self._save_summary()
