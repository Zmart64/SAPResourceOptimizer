"""
Simple example script for running SizeyPredictor with preprocessed data.

This script demonstrates how to:
1. Load preprocessed data
2. Initialize and configure SizeyPredictor
3. Train the model
4. Make predictions
5. Evaluate performance
"""

import logging
import pickle
import warnings
from pathlib import Path

# Import the SizeyPredictor
from resource_prediction.models.implementations import SizeyPredictor
from resource_prediction.models.implementations.sizey import OffsetStrategy
from resource_prediction.reporting import calculate_allocation_categories

# Import business evaluation functions
from resource_prediction.training.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def main():
    """Main function to run all demonstrations."""
    print("SizeyPredictor Demo with Preprocessed Data")
    print("=" * 60)

    try:
        data_path = Path("data/processed")

        # Load training data
        with open(data_path / "X_train.pkl", "rb") as f:
            x_train = pickle.load(f)[:100]
        with open(data_path / "y_train.pkl", "rb") as f:
            y_train = pickle.load(f)[:100]

        # Load test data
        with open(data_path / "X_test.pkl", "rb") as f:
            x_test = pickle.load(f)[:100]
        with open(data_path / "y_test.pkl", "rb") as f:
            y_test = pickle.load(f)[:100]

        print(f"Full dataset: x_train={x_train.shape}, y_train={y_train.shape}")

        print("Creating predictor")
        sizey = SizeyPredictor(offset_strat=OffsetStrategy.STD_ALL, use_softmax=False)
        print("Predictor created successfully!")

        print("Training Sizey model...")
        sizey.fit(x_train, y_train)
        print("Model trained successfully!")

        # Make predictions on test subset
        print("Making predictions on test subset")
        y_test_subset = y_test
        x_test_subset = x_test

        print(
            f"Test subset: x_test={x_test_subset.shape}, y_test={y_test_subset.shape}"
        )

        predictions = sizey.predict(x_test_subset, y_test_subset)
        # Convert predictions to numpy array for compatibility with reporting functions

        # Ensure y_test_subset.values is 1D for compatibility with reporting functions
        y_true_values = y_test_subset.values.flatten()

        # Use business evaluation functions from resource_prediction
        print("Evaluation Results:")
        print("-" * 60)

        # Calculate business metrics using the same functions as the main pipeline
        business_metrics = Trainer._allocation_metrics(predictions, y_true_values)
        business_score = Trainer._business_score(business_metrics)

        print(f"Under-allocation Rate: {business_metrics['under_pct']:.2f}%")
        print(f"Mean GB Wasted: {business_metrics['mean_gb_wasted']:.2f} GB")
        print(f"Total Over-allocation: {business_metrics['total_over_pct']:.2f}%")
        print(f"Business Score (Lower is Better): {business_score:.2f}")

        # Calculate detailed allocation categories for business reporting
        allocation_stats = calculate_allocation_categories(
            name="SizeyPredictor",
            allocations=predictions,
            true_values=y_true_values,
        )

        print("\n Detailed Business Allocation Analysis:")
        print("-" * 60)
        print(
            f"Under-allocated jobs: {allocation_stats['Under-allocated_jobs']} ({allocation_stats['Under-allocated_perc']:.1f}%)"
        )
        print(
            f"Perfectly-allocated jobs: {allocation_stats['Perfectly-allocated_jobs']} ({allocation_stats['Perfectly-allocated_perc']:.1f}%)"
        )
        print(
            f"Well-allocated (1x-2x): {allocation_stats['Well-allocated (1x-2x)_jobs']} ({allocation_stats['Well-allocated (1x-2x)_perc']:.1f}%)"
        )
        print(
            f"Severely Over (2x-3x): {allocation_stats['Severely Over (2x-3x)_jobs']} ({allocation_stats['Severely Over (2x-3x)_perc']:.1f}%)"
        )
        print(
            f"Extremely Over (3x-4x): {allocation_stats['Extremely Over (3x-4x)_jobs']} ({allocation_stats['Extremely Over (3x-4x)_perc']:.1f}%)"
        )
        print(
            f"Massively Over (4x+): {allocation_stats['Massively Over (4x+)_jobs']} ({allocation_stats['Massively Over (4x+)_perc']:.1f}%)"
        )

        print("\n Business Impact Summary:")
        print("-" * 60)
        print(
            f"Total true memory needed: {allocation_stats['total_true_mem_gb']:.2f} GB"
        )
        print(
            f"Total allocated memory: {allocation_stats['total_alloc_mem_gb']:.2f} GB"
        )
        print(
            f"Total over-allocated memory: {allocation_stats['total_over_alloc_gb']:.2f} GB"
        )
        print(
            f"Total under-allocated memory: {allocation_stats['total_under_alloc_gb']:.2f} GB"
        )

        # # Display sample-by-sample results
        # print("\n📋 Sample-by-Sample Analysis:")
        # print("-" * 60)
        # for i, _ in enumerate(predictions):
        #     actual = float(y_test_subset.iloc[i])
        #     predicted = predictions[i]
        #     error_pct = abs(predicted - actual) / actual * 100

        #     # Determine allocation category for business context
        #     if predicted < actual:
        #         status = "⚠️  UNDER-ALLOCATED"
        #     elif predicted == actual:
        #         status = "✅ PERFECT"
        #     elif predicted < actual * 2:
        #         status = "🟢 WELL-ALLOCATED"
        #     elif predicted < actual * 3:
        #         status = "🟡 SEVERELY OVER"
        #     elif predicted < actual * 4:
        #         status = "🟠 EXTREMELY OVER"
        #     else:
        #         status = "🔴 MASSIVELY OVER"

        #     print(
        #         f"Sample {i + 1:2d}: "
        #         f"Actual={actual:8.2f} GB, "
        #         f"Predicted={predicted:8.2f} GB, "
        #         f"Error={error_pct:5.1f}% "
        #         f"{status}"
        #     )

        print("\n🎯 Business Evaluation Summary:")
        print("-" * 60)
        print(f"Business Score: {business_score:.2f} (Lower = Better)")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
