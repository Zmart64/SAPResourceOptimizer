import pandas as pd
import numpy as np
import ast
import xgboost
import lightgbm
import catboost
from sklearn.cluster import KMeans

from resource_prediction.config import Config
from resource_prediction.training.hyperparameter import BayesianOptimizer


class Trainer:
    """Orchestrates model optimization and final evaluation on the holdout test set."""

    def __init__(self, config: Config):
        self.config = config

    def run_bayesian_optimization(self):
        """Runs optimization using ONLY the training data."""
        print("--- Loading Training Data for Optimization ---")
        try:
            X_train = pd.read_pickle(self.config.X_TRAIN_PATH)
            y_train = pd.read_pickle(self.config.Y_TRAIN_PATH)
        except FileNotFoundError:
            print(
                "Error: Training data not found. Please run preprocessing first (without --skip-preprocessing).")
            return

        optimizer = BayesianOptimizer(
            config=self.config, X_train=X_train, y_train=y_train)
        optimizer.run()

    def evaluate_best_model_on_test_set(self):
        """
        Loads the best model from the optimization summary, trains it on the
        full training set, and evaluates it on the holdout test set.
        """
        print("\n--- Starting Final Evaluation on Holdout Test Set ---")

        try:
            summary = pd.read_csv(
                self.config.OPTIMIZATION_SUMMARY_PATH).iloc[0]
            best_model_name = summary['model_name']
            best_params = ast.literal_eval(summary['best_params_from_grid'])
        except (FileNotFoundError, IndexError):
            print(
                f"Error: Could not load optimization summary from {self.config.OPTIMIZATION_SUMMARY_PATH}")
            print("Please run the optimization first with '--run-optimization'.")
            return

        print("--- Loading all data splits for final evaluation ---")
        X_train = pd.read_pickle(self.config.X_TRAIN_PATH)
        y_train = pd.read_pickle(self.config.Y_TRAIN_PATH)
        X_test = pd.read_pickle(self.config.X_TEST_PATH)
        y_test = pd.read_pickle(self.config.Y_TEST_PATH)

        y_train_gb = y_train[self.config.TARGET_COLUMN_PROCESSED]
        y_test_gb = y_test[self.config.TARGET_COLUMN_PROCESSED]

        print("--- Re-creating binning strategy from training data ---")
        n_bins, strategy = best_params['n_bins'], best_params['strategy']
        min_val, max_val = y_train_gb.min(), y_train_gb.max()
        if strategy == 'quantile':
            _, bin_edges = pd.qcut(y_train_gb, q=n_bins,
                                   retbins=True, duplicates='drop')
        elif strategy == 'uniform':
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        else:
            kmeans = KMeans(n_clusters=n_bins, random_state=self.config.RANDOM_STATE, n_init='auto').fit(
                y_train_gb.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            edges = [(centers[i] + centers[i+1]) /
                     2 for i in range(len(centers)-1)]
            bin_edges = np.array([min_val] + edges + [max_val])
        bin_edges, num_classes = sorted(
            list(set(bin_edges))), len(bin_edges) - 1
        y_train_binned = pd.cut(y_train_gb, bins=bin_edges, labels=range(
            num_classes), right=True, include_lowest=True)

        print(
            f"--- Training final {best_model_name.upper()} model on {len(X_train)} samples ---")
        model_hyperparams = {k: v for k, v in best_params.items(
        ) if k not in ['n_bins', 'strategy', 'confidence_threshold']}

        if best_model_name == 'xgboost':
            model = xgboost.XGBClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, n_jobs=-1)
        elif best_model_name == 'lightgbm':
            model = lightgbm.LGBMClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            model = catboost.CatBoostClassifier(
                **model_hyperparams, random_state=self.config.RANDOM_STATE, verbose=0, thread_count=-1)

        model.fit(X_train, y_train_binned)

        print(f"--- Evaluating model on {len(X_test)} unseen test samples ---")
        y_pred_probs = model.predict_proba(X_test)
        y_pred_base = np.argmax(y_pred_probs, axis=1)
        confidence_threshold = best_params['confidence_threshold']
        y_pred_final = [min(pred + 1, num_classes - 1) if conf < confidence_threshold else pred
                        for pred, conf in zip(y_pred_base, y_pred_probs[np.arange(len(y_pred_base)), y_pred_base])]

        allocated_mem_pred = np.array(
            [bin_edges[min(c + 1, num_classes)] for c in y_pred_final])
        true_mem_test = y_test_gb.values

        total_jobs, jobs_under = len(true_mem_test), np.sum(
            allocated_mem_pred < true_mem_test)
        over_allocation_gb = np.sum(np.maximum(
            0, allocated_mem_pred - true_mem_test))
        total_true_used_gb = np.sum(true_mem_test)

        final_metrics = {
            'perc_under_allocation': (jobs_under / total_jobs) * 100,
            'perc_total_over_allocation': (over_allocation_gb / total_true_used_gb) * 100 if total_true_used_gb > 0 else 0,
        }

        print("\n--- Final Test Set Performance ---")
        print(
            f"Percentage of jobs with under-allocated memory: {final_metrics['perc_under_allocation']:.2f}%")
        print(
            f"Percentage of total over-allocated memory: {final_metrics['perc_total_over_allocation']:.2f}%")

        report_df = pd.DataFrame([final_metrics])
        report_df['model_name'] = best_model_name
        report_df['best_params'] = str(best_params)
        report_df.to_csv(self.config.FINAL_EVALUATION_REPORT_PATH, index=False)
        print(
            f"\nFinal evaluation report saved to {self.config.FINAL_EVALUATION_REPORT_PATH.resolve()}")
