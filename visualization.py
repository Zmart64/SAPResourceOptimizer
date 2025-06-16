"""
Create plots and visualizations
"""

import matplotlib.pyplot as plt


class Visualizer:
    """Visualization class for model evaluation"""

    def __init__(self, results):
        self.results = results

    def plot_predictions_vs_actual(self, model_name, output_dir):
        """Plot predicted vs actual values for regression evaluation"""
        if model_name.lower() == "random forest":
            results = self.results.get("random_forest")
        elif model_name.lower() == "xgboost":
            results = self.results.get("xgboost")
        else:
            print(f"Model {model_name} not found in results")
            return

        if not results:
            print(f"No results found for {model_name}")
            return

        y_test = self.results["test_data"]["y_test"]
        predictions = results["predictions"]

        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, predictions, alpha=0.6, edgecolors="k", linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        plt.xlabel("Actual Max RSS (MB)")
        plt.ylabel("Predicted Max RSS (MB)")
        plt.title(f"{model_name} - Predicted vs Actual Memory Usage")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add R² score on plot
        r2 = results["r2"]
        plt.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_predictions_vs_actual.png")

    def plot_predictions_over_time(self, model_name, output_dir):
        """Plot model predictions over time against actual values"""
        if model_name.lower() == "random forest":
            results = self.results.get("random_forest")
        elif model_name.lower() == "xgboost":
            results = self.results.get("xgboost")
        else:
            print(f"Model {model_name} not found in results")
            return

        if not results:
            print(f"No results found for {model_name}")
            return

        y_test = self.results["test_data"]["y_test"]
        predictions = results["predictions"]

        plt.figure(figsize=(14, 8))

        # Plot actual values
        plt.plot(
            range(len(y_test)), y_test, "b-", alpha=0.7, label="Actual", linewidth=1.5
        )

        # Plot predictions
        plt.plot(
            range(len(predictions)),
            predictions,
            "r--",
            alpha=0.8,
            label="Predicted",
            linewidth=1.5,
        )

        plt.xlabel("Time Index")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"{model_name} - Predictions vs Actual Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add R² score
        r2 = results["r2"]
        plt.text(
            0.02,
            0.98,
            f"R² = {r2:.4f}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="top",
        )

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_predictions_over_time.png")
        plt.close()
