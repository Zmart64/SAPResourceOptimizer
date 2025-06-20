"""
Model Training
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config


class ModelTrainer:
    """Model Trainer class for max_rss prediction"""

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.feature_columns = None
        self.results = {}

    def split_data(
        self, x: pd.DataFrame, y: pd.Series, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets (time-aware split)"""
        # Calculate split index based on test_size
        split_index = int(len(x) * (1 - test_size))

        # Split chronologically
        x_train = x.iloc[:split_index]
        x_test = x.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        print(f"Training set: {x_train.shape}")
        print(f"Test set: {x_test.shape}")
        return x_train, x_test, y_train, y_test

    def train_random_forest(self, x_train, y_train):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        self.rf_model = RandomForestRegressor(**Config.RANDOM_FOREST_PARAMS.copy())
        self.rf_model.fit(x_train, y_train)

        print("Random Forest training complete!")
        return self.rf_model

    def train_xgboost(self, x_train, y_train):
        """Train XGBoost model"""

        print("\nTraining XGBoost...")
        self.xgb_model = xgb.XGBRegressor(**Config.XGBOOST_PARAMS.copy())
        self.xgb_model.fit(x_train, y_train)

        print("XGBoost training complete!")
        return self.xgb_model

    # def train_xgboost(self, x_train, y_train, **kwargs):
    #     """Train XGBoost model with quantile regression for overestimation"""

    #     print("\nTraining XGBoost with quantile regression...")

    #     # Default parameters with quantile objective
    #     params = {
    #         "n_estimators": 100,
    #         "max_depth": 6,
    #         "learning_rate": 0.1,
    #         "random_state": 42,
    #         "n_jobs": -1,
    #         "objective": "reg:quantileerror",
    #         "quantile_alpha": 0.8,  # Predict 80th percentile (adjust 0.7-0.9)
    #     }
    #     params.update(kwargs)

    #     self.xgb_model = xgb.XGBRegressor(**params)
    #     self.xgb_model.fit(x_train, y_train)

    #     print("XGBoost quantile training complete!")
    #     return self.xgb_model

    def evaluate_model(self, model, x_test, y_test, model_name):
        """Evaluate model performance"""
        predictions = model.predict(x_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)

        results = {
            "model_name": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mse": mse,
            "predictions": predictions,
        }

        print(f"{model_name} Results:")
        print(f"  RMSE: {rmse:.2f} MB")
        print(f"  MAE: {mae:.2f} MB")
        print(f"  R²: {r2:.6f}")
        print(f"  Accuracy: {r2 * 100:.4f}%")

        return results

    def get_feature_importance(self, model, feature_columns, model_name):
        """Get feature importance from trained model"""
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {"feature": feature_columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            print(f"\nTop 10 Most Important Features ({model_name}):")
            print(importance_df.head(10))

            return importance_df
        else:
            print(f"Model {model_name} does not support feature importance")
            return None

    def train_all_models(self, x, y, feature_columns, test_size):
        """Train all available models and evaluate them"""
        print("=== Training All Models ===")

        # Add this debug information
        print("Training dataset info:")
        print(f"  Features shape: {x.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Memory usage: ~{x.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if len(x) < 100:
            print("WARNING: Training dataset is very small!")

        self.feature_columns = feature_columns

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = self.split_data(x, y, test_size=test_size)

        # Train Random Forest
        rf_model = self.train_random_forest(x_train, y_train)
        rf_results = self.evaluate_model(rf_model, x_test, y_test, "Random Forest")
        rf_importance = self.get_feature_importance(
            rf_model, feature_columns, "Random Forest"
        )

        self.results["random_forest"] = rf_results
        self.results["random_forest"]["importance"] = rf_importance

        # Train XGBoost
        xgb_model = self.train_xgboost(x_train, y_train)
        xgb_results = self.evaluate_model(xgb_model, x_test, y_test, "XGBoost")
        xgb_importance = self.get_feature_importance(
            xgb_model, feature_columns, "XGBoost"
        )

        self.results["xgboost"] = xgb_results
        self.results["xgboost"]["importance"] = xgb_importance

        # Store test data for plotting
        self.results["test_data"] = {"X_test": x_test, "y_test": y_test}

        print("=== Model Training Complete ===\n")
        return self.results

    def save_models(self, save_dir):
        """Save trained models"""

        os.makedirs(save_dir, exist_ok=True)

        if self.rf_model:
            joblib.dump(self.rf_model, f"{save_dir}/random_forest_model.pkl")
            print(f"Random Forest model saved to {save_dir}/random_forest_model.pkl")

        if self.xgb_model:
            joblib.dump(self.xgb_model, f"{save_dir}/xgboost_model.pkl")
            print(f"XGBoost model saved to {save_dir}/xgboost_model.pkl")

        # Save feature columns
        # joblib.dump(self.feature_columns, f"{save_dir}/feature_columns.pkl")
        # print(f"Feature columns saved to {save_dir}/feature_columns.pkl")

    def predict(self, X, model_type="random_forest"):
        """Make predictions using trained model"""
        if model_type == "random_forest" and self.rf_model:
            return self.rf_model.predict(X)
        elif model_type == "xgboost" and self.xgb_model:
            return self.xgb_model.predict(X)
        else:
            raise ValueError(f"Model {model_type} not available or not trained")
