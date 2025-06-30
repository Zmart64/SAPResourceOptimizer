"""
Model Training (Refactored for Classification, CV, and Saving on a Subset of Data)
"""

import os
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             make_scorer, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# Import the updated config
from config import Config

# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelTrainer:
    """Model Trainer class for max_rss classification and prediction"""

    def __init__(self, config: Config):
        self.config = config
        self.xgb_model = None
        self.feature_columns = None
        self.bin_edges_gb = None
        self.class_labels = None
        self.best_params = None
        self.results = {}

        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        print(
            f"Output will be saved to: {os.path.abspath(self.config.OUTPUT_DIR)}")

    def _split_build_profile(self, profile_string):
        """Helper to split buildProfile string into components."""
        if not isinstance(profile_string, str):
            return pd.Series(["unknown", "unknown", "unknown"])
        parts = profile_string.split('-')
        arch = parts[0]
        compiler = "unknown"
        opt = "unknown"
        if len(parts) > 1:
            compiler = parts[1]
        if len(parts) > 2:
            opt = parts[2]
        return pd.Series([arch, compiler, opt])

    # MODIFICATION: This method now accepts a DataFrame instead of a filepath
    def preprocess_data(self, train_df: pd.DataFrame):
        """Processes the provided training dataframe, creating features and the target variable."""
        print("\n--- 1. Preprocessing Training Data ---")

        # Feature Engineering on the provided dataframe
        df = train_df.copy()
        df[self.config.TARGET_COLUMN] = pd.to_numeric(
            df[self.config.TARGET_COLUMN], errors='coerce')
        df.dropna(subset=[self.config.TARGET_COLUMN], inplace=True)
        df['max_rss_gb'] = df[self.config.TARGET_COLUMN] / (1024**3)

        F = df.copy()
        F["ts_month"] = F["time"].dt.month
        F["ts_dow"] = F["time"].dt.dayofweek
        F["ts_hour"] = F["time"].dt.hour
        F[["bp_arch", "bp_compiler", "bp_opt"]] = F["buildProfile"].apply(
            self._split_build_profile)
        F["branch_prefix"] = F["branch"].str.replace(
            r"[\d_]*$", "", regex=True).replace('', 'unknown_prefix')
        F["target_cnt"] = F["targets"].astype(str).str.count(",") + 1

        categorical_features = ["location", "branch_prefix", "bp_arch", "bp_compiler",
                                "bp_opt", "makeType", "component", "ts_month", "ts_dow", "ts_hour"]
        numerical_features = ["jobs", "target_cnt"]

        features_to_use = sorted(list(set(F.columns) & set(
            categorical_features + numerical_features)))
        X = F[features_to_use].copy()

        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype(str).astype('category')

        # Target Variable Creation
        self.bin_edges_gb = self.config.MANUAL_BIN_EDGES_GB
        num_target_classes = len(self.bin_edges_gb) - 1
        self.class_labels = list(range(num_target_classes))

        y = pd.cut(F['max_rss_gb'], bins=self.bin_edges_gb, labels=self.class_labels,
                   right=False, include_lowest=True, duplicates='drop')

        # Final Cleaning
        valid_indices = y.dropna().index
        X, y, F_eval = X.loc[valid_indices], y.loc[valid_indices], F.loc[valid_indices]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(int)
        F_eval = F_eval.reset_index(drop=True)

        self.feature_columns = X.columns.tolist()
        print(
            f"Preprocessing complete. Final training shapes: X={X.shape}, y={y.shape}")
        print("Class distribution in training data:\n",
              y.value_counts().sort_index())

        return X, y, F_eval

    # ... (the rest of the ModelTrainer class remains exactly the same) ...
    # ... (_weighted_recall_scorer_func, find_best_params_with_gridsearch, etc.) ...
    def _weighted_recall_scorer_func(self, y_true, y_pred, labels, class_weights_for_recall):
        recalls = recall_score(y_true, y_pred, labels=labels,
                               average=None, zero_division=0)
        return np.sum(recalls * class_weights_for_recall) / np.sum(class_weights_for_recall)

    def find_best_params_with_gridsearch(self, X, y):
        print("\n--- 2. Finding Best Hyperparameters with GridSearchCV ---")
        tscv = TimeSeriesSplit(n_splits=self.config.N_SPLITS_CV)

        if len(self.config.RECALL_WEIGHTS) != len(self.class_labels):
            raise ValueError(
                "RECALL_WEIGHTS in Config must have the same length as the number of classes.")

        custom_scorer = make_scorer(self._weighted_recall_scorer_func, greater_is_better=True,
                                    labels=self.class_labels, class_weights_for_recall=self.config.RECALL_WEIGHTS)

        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.class_labels),
            eval_metric='mlogloss',
            random_state=self.config.RANDOM_STATE,
            enable_categorical=True,
            n_jobs=1
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.config.PARAM_GRID,
            scoring=custom_scorer,
            cv=tscv,
            verbose=0,
            n_jobs=self.config.GRID_SEARCH_N_JOBS
        )

        num_fits = np.prod(
            [len(v) for v in self.config.PARAM_GRID.values()]) * self.config.N_SPLITS_CV
        print(
            f"Starting GridSearchCV with {num_fits} total fits (on the {self.config.TRAIN_DATA_RATIO*100}% training set)...")

        try:
            with tqdm_joblib(tqdm(total=num_fits, desc="GridSearch Progress")) as progress_bar:
                grid_search.fit(X, y)
            self.best_params = grid_search.best_params_
            print(
                f"GridSearchCV complete. Best Score (Weighted Recall): {grid_search.best_score_:.4f}")
            print(f"Best parameters found: {self.best_params}")
        except Exception as e:
            print(
                f"ERROR during GridSearchCV: {e}. Using default params from grid.")
            self.best_params = {k: v[0]
                                for k, v in self.config.PARAM_GRID.items()}

        return self.best_params

    def train_and_evaluate_cv(self, X, y, F_eval):
        print("\n--- 3. Evaluating Final Model with Cross-Validation ---")
        if self.best_params is None:
            raise ValueError(
                "Best parameters not set. Run find_best_params_with_gridsearch first.")

        final_model_params = {
            'objective': 'multi:softmax',
            'num_class': len(self.class_labels),
            'eval_metric': 'mlogloss',
            'random_state': self.config.RANDOM_STATE,
            'enable_categorical': True,
            'n_jobs': -1,
            **self.best_params
        }

        eval_model = xgb.XGBClassifier(**final_model_params)
        tscv = TimeSeriesSplit(n_splits=self.config.N_SPLITS_CV)
        training_weights_map = {label: (
            idx + 1)**self.config.TRAINING_CLASS_WEIGHTS_EXPONENT for idx, label in enumerate(self.class_labels)}

        all_y_true, all_y_pred = [], []

        for fold, (train_index, test_index) in enumerate(tscv.split(X, y)):
            print(f"--- Fold {fold+1}/{self.config.N_SPLITS_CV} ---")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            sample_weights = np.array(
                [training_weights_map.get(label, 1) for label in y_train])
            eval_model.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = eval_model.predict(X_test)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            print(classification_report(y_test, y_pred, labels=self.class_labels,
                  target_names=[f"C{l}" for l in self.class_labels], zero_division=0))

        self.results['all_y_true'] = all_y_true
        self.results['all_y_pred'] = all_y_pred
        self._calculate_final_metrics()

    def _calculate_final_metrics(self):
        print("\n--- Aggregate CV Results (on Training Data) ---")
        report_str = classification_report(self.results['all_y_true'], self.results['all_y_pred'], labels=self.class_labels, target_names=[
                                           f"C{l}" for l in self.class_labels], zero_division=0)
        print(report_str)
        self.results['classification_report'] = report_str
        cm = confusion_matrix(
            self.results['all_y_true'], self.results['all_y_pred'], labels=self.class_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                    f"C{l}" for l in self.class_labels], yticklabels=[f"C{l}" for l in self.class_labels])
        plt.title('Aggregated Confusion Matrix (All Folds on Training Data)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(self.config.OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.show()
        print(f"Confusion matrix saved to {cm_path}")

    def train_final_model(self, X, y):
        print("\n--- 4. Training Final Model on ENTIRE Training Set ---")
        if self.best_params is None:
            raise ValueError(
                "Best parameters not set. Run find_best_params_with_gridsearch first.")

        final_model_params = {'objective': 'multi:softmax', 'num_class': len(
            self.class_labels), 'random_state': self.config.RANDOM_STATE, 'enable_categorical': True, 'n_jobs': -1, **self.best_params}
        self.xgb_model = xgb.XGBClassifier(**final_model_params)
        training_weights_map = {label: (
            idx + 1)**self.config.TRAINING_CLASS_WEIGHTS_EXPONENT for idx, label in enumerate(self.class_labels)}
        sample_weights = np.array(
            [training_weights_map.get(label, 1) for label in y])
        self.xgb_model.fit(X, y, sample_weight=sample_weights)
        print(
            f"Final model has been trained on all {len(X)} samples of the training data.")
        plt.figure(figsize=(12, 8))
        importances = self.xgb_model.feature_importances_
        sorted_indices = np.argsort(importances)
        plt.barh(np.array(self.feature_columns)[
                 sorted_indices], importances[sorted_indices])
        plt.xlabel("XGBoost Feature Importance")
        plt.title("Feature Importance (Final Model)")
        plt.tight_layout()
        fi_path = os.path.join(self.config.OUTPUT_DIR,
                               "feature_importance.png")
        plt.savefig(fi_path)
        plt.show()
        print(f"Feature importance plot saved to {fi_path}")

    def save_model(self):
        if self.xgb_model is None:
            print("No model has been trained. Cannot save.")
            return

        model_path = os.path.join(
            self.config.OUTPUT_DIR, self.config.MODEL_FILENAME)
        model_payload = {'model': self.xgb_model, 'feature_columns': self.feature_columns,
                         'bin_edges_gb': self.bin_edges_gb, 'class_labels': self.class_labels}
        joblib.dump(model_payload, model_path)
        print(
            f"\n--- 5. Model Saved ---\nModel payload saved to: {model_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    config = Config()
    trainer = ModelTrainer(config)

    # Load the full dataset and split it first
    print(f"Loading full dataset from {config.FULL_DATA_FILE}...")
    try:
        full_df = pd.read_csv(config.FULL_DATA_FILE, sep=";")
    except FileNotFoundError:
        print(f"Error: Data file '{config.FULL_DATA_FILE}' not found.")
        exit()

    # Sort by time to ensure chronological split
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df = full_df.sort_values('time').reset_index(drop=True)

    # Perform the 80/20 split based on the config
    split_index = int(len(full_df) * config.TRAIN_DATA_RATIO)
    train_df = full_df.iloc[:split_index]
    simulate_df = full_df.iloc[split_index:]  # This part is held out

    print(f"Full dataset size: {len(full_df)}")
    print(
        f"Splitting data: {len(train_df)} for training ({config.TRAIN_DATA_RATIO*100}%), {len(simulate_df)} for simulation.")

    simulation_data_path = os.path.join(
        config.OUTPUT_DIR, "simulation_data.csv")
    simulate_df.to_csv(simulation_data_path, index=False, sep=";")
    print(f"Simulation data saved to: {simulation_data_path}")

    # Pass ONLY the training data to the pipeline
    X_train, y_train, F_eval_train = trainer.preprocess_data(train_df)

    if X_train is not None and y_train is not None:
        # 2. Find best hyperparameters using the training data
        best_params = trainer.find_best_params_with_gridsearch(
            X_train, y_train)

        # 3. Evaluate the model with CV, also only on the training data
        trainer.train_and_evaluate_cv(X_train, y_train, F_eval_train)

        # 4. Train one final model on ALL of the training data
        trainer.train_final_model(X_train, y_train)

        # 5. Save the final model and metadata
        trainer.save_model()

        print("\nScript finished successfully.")
