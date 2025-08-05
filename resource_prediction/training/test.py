import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

# Import the configuration class to get access to all the file paths
from resource_prediction.config import Config


def generate_report():
    """
    Loads completed evaluation results from CSV files and generates a
    comparison bar chart of the hold-out set performance.
    """
    print("Starting report generation...")
    config = Config()

    # 1. Prepare to load data
    all_dfs = []
    regression_csv = config.REGRESSION_RESULTS_CSV_PATH
    classification_csv = config.CLASSIFICATION_RESULTS_CSV_PATH

    # 2. Robustly load regression results if the file exists
    if regression_csv.exists():
        print(f"Loading regression data from: {regression_csv}")
        df_regr = pd.read_csv(regression_csv)
        all_dfs.append(df_regr)
    else:
        print(
            f"Warning: Regression results file not found at {regression_csv}")

    # 3. Robustly load classification results if the file exists
    if classification_csv.exists():
        print(f"Loading classification data from: {classification_csv}")
        df_class = pd.read_csv(classification_csv)
        all_dfs.append(df_class)
    else:
        print(
            f"Warning: Classification results file not found at {classification_csv}")

    # 4. Check if any data was loaded
    if not all_dfs:
        print("Error: No result files found in the 'output' directory. Nothing to plot.")
        return

    # 5. Combine data and prepare for plotting
    all_results = pd.concat(all_dfs, ignore_index=True)

    if 'score_hold' not in all_results.columns:
        print("Error: 'score_hold' column not found in the result files. Cannot generate plot.")
        return

    print(f"\nFound {len(all_results)} models to plot.")

    # 6. Check for large outliers and decide on scale
    use_log_scale = False
    if all_results['score_hold'].max() > 10 * all_results['score_hold'].median():
        print("Warning: A large outlier was detected in the scores. Using a logarithmic scale for better visualization.")
        use_log_scale = True

    print("Generating comparison chart...")

    # 7. Create and save the plot
    plt.figure(figsize=(10, 8))

    order = all_results.sort_values("score_hold")["model"]

    sns.barplot(
        data=all_results,
        y="model",
        x="score_hold",
        order=order,
        color="steelblue"
    )

    # This is the fix: Apply a log scale to the x-axis if needed
    if use_log_scale:
        plt.xscale('log')

    plt.xlabel("Hold-out Set Business Score (Lower is Better)")
    plt.ylabel("Model Architecture")
    plt.title("Final Model Performance on Hold-out Data")
    plt.tight_layout()

    output_path = config.RESULTS_PLOT_PATH
    plt.savefig(output_path)

    print(f"\nComparison chart successfully saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a comparison chart from existing evaluation result CSV files in the 'output' directory."
    )
    args = parser.parse_args()

    generate_report()
