"""
Main entrypoint, implements the complete pipeline.
"""

import argparse
import os
from datetime import datetime

from config import Config
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from visualization import Visualizer


def create_directories():
    """Create necessary directories for outputs"""
    directories = [Config.RESULTS_DIR, Config.MODELS_DIR, Config.PLOTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def print_header():
    """Print a nice header for the script"""
    print("=" * 70)
    print("MAX_RSS memory usage prediction pipeline")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def summerize_results(results, df_clean, file_path):
    """Save results summary to a text file"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("MAX_RSS memory usage prediction pipeline - RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATASET OVERVIEW:\n")
        f.write(f"Processed records: {len(df_clean):,}\n")
        f.write(f"Features: {len(results.get('feature_columns', []))}\n\n")

        f.write("MODEL PERFORMANCE:\n")
        best_model = None
        best_r2 = 0

        for model_name, result in results.items():
            if model_name != "test_data":
                r2 = result["r2"]
                model_display_name = result["model_name"]
                f.write(f"\n{model_display_name}:\n")
                f.write(f"  RMSE: {result['rmse']:.2f} MB\n")
                f.write(f"  MAE: {result['mae']:.2f} MB\n")
                f.write(f"  R² Score: {r2:.6f}\n")

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_display_name

        f.write("\nKEY INSIGHTS:")
        f.write(f"Best performing model: {best_model}")
        f.write(f"Best R² score: {best_r2:.6f}")

        # if "random_forest" in results and "importance" in results["random_forest"]:
        #     f.write("\nTOP 10 FEATURE IMPORTANCE (Random Forest):\n")
        #     importance_df = results["random_forest"]["importance"]
        #     for _, row in importance_df.head(10).iterrows():
        #         f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

    print(f"Results summary saved to: {file_path}")
    print(f"Models saved in: {file_path}/models/")
    print(f"Plots saved in: {file_path}/plots/")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Max_rss Prediction Pipeline")
    parser.add_argument(
        "--preprocessing-only",
        action="store_true",
        help="Run only data preprocessing step",
    )
    parser.add_argument(
        "--data-path",
        default=Config.DATA_PATH,
        help="Path to the data file",
    )

    args = parser.parse_args()

    try:
        print_header()
        create_directories()

        if not os.path.exists(args.data_path):
            print(f"Error: Data file not found at {args.data_path}")
            print("Please ensure the build data file is located at the correct path.")
            return

        print(f"Data file found: {args.data_path}")

        # Step 1: Data Preprocessing
        print("\nDATA PREPROCESSING")
        print("-" * 40)

        preprocessor = DataPreprocessor()
        x, y, feature_columns, df_clean = preprocessor.preprocess_pipeline(
            args.data_path
        )

        if args.preprocessing_only:
            return

        # Step 2: Model Training
        print("\nMODEL TRAINING")
        print("-" * 40)
        trainer = ModelTrainer()

        results = trainer.train_all_models(
            x,
            y,
            feature_columns,
            test_size=Config.TEST_SIZE,
        )

        # Save models
        trainer.save_models(Config.MODELS_DIR)

        # Step 3: Visualization
        print("\nCREATING VISUALIZATIONS")
        print("-" * 40)
        visualizer = Visualizer(results=results)
        visualizer.plot_predictions_vs_actual("Random Forest", Config.PLOTS_DIR)
        visualizer.plot_predictions_vs_actual("XGboost", Config.PLOTS_DIR)

        # visualizer.plot_predictions_over_time("Random Forest", Config.PLOTS_DIR)
        # visualizer.plot_predictions_over_time("XGboost", Config.PLOTS_DIR)

        # Step 4: Generate Summary Report
        print("\nGENERATING SUMMARY REPORT")
        print("-" * 40)
        results_path = f"{Config.RESULTS_DIR}/summary_report.txt"
        summerize_results(results, df_clean, results_path)

        print("\nPipeline completed successfully!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check that all required files are in the correct locations.")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred: {e}")
        print("Please check the error message and try again.")


if __name__ == "__main__":
    main()
