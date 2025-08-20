"""Functions for generating detailed model evaluation reports and plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    "Under-allocated": "#e6261f",         # Red
    "Perfectly-allocated": "#64a85f",      # Darker Green
    "Well-allocated (1x-2x)": "#99c193",  # Light Green
    "Severely Over (2x-3x)": "#fde940",   # Pale Yellow
    "Extremely Over (3x-4x)": "#f5b70d",  # Golden Yellow
    "Massively Over (4x+)": "#f37f00",    # Orange
}
LABELS = list(COLORS.keys())


def calculate_allocation_categories(
    name: str,
    allocations: np.ndarray,
    true_values: np.ndarray,
    under_allocation_mask: np.ndarray | None = None
) -> dict:
    """
    Calculates the distribution of jobs into detailed allocation categories
    using a vectorized approach for performance.

    Args:
        name (str): The name of the model or 'Baseline'.
        allocations (np.ndarray): The memory allocated for each job (GB).
        true_values (np.ndarray): The true memory used by each job (GB).
        under_allocation_mask (np.ndarray | None): A boolean array where True
            indicates a job is definitively under-allocated (e.g., from fail counts).

    Returns:
        dict: A dictionary containing detailed statistics for the allocation strategy.
    """
    total_jobs = len(true_values)
    if total_jobs == 0:
        return {}

    true_values_safe = np.maximum(true_values, 1e-9)
    counts = {label: 0 for label in LABELS}

    if under_allocation_mask is None:
        is_under = allocations < true_values_safe
    else:
        is_under = under_allocation_mask | (allocations < true_values_safe)

    counts["Under-allocated"] = np.sum(is_under)

    # Process jobs that were not under-allocated
    not_under_mask = ~is_under
    alloc_ok = allocations[not_under_mask]
    true_ok = true_values[not_under_mask]

    # Perfectly allocated jobs
    is_perfect = np.isclose(alloc_ok, true_ok)
    counts["Perfectly-allocated"] = np.sum(is_perfect)

    # Over-allocated jobs (those not under and not perfect)
    is_over_mask = ~is_perfect
    alloc_over = alloc_ok[is_over_mask]
    true_over = true_ok[is_over_mask]
    true_over_safe = np.maximum(true_over, 1e-9)

    ratio = alloc_over / true_over_safe
    counts["Well-allocated (1x-2x)"] = np.sum((ratio >= 1) & (ratio < 2))
    counts["Severely Over (2x-3x)"] = np.sum((ratio >= 2) & (ratio < 3))
    counts["Extremely Over (3x-4x)"] = np.sum((ratio >= 3) & (ratio < 4))
    counts["Massively Over (4x+)"] = np.sum(ratio >= 4)

    # Prepare detailed report dictionary
    report = {"model_name": name}
    for label in LABELS:
        count = counts[label]
        perc = (count / total_jobs) * 100
        report[f"{label}_jobs"] = count
        report[f"{label}_perc"] = perc

    over_alloc_mem = np.maximum(0, allocations - true_values)
    under_alloc_mem = np.maximum(0, true_values - allocations)
    report["total_jobs"] = total_jobs
    report["total_true_mem_gb"] = np.sum(true_values)
    report["total_alloc_mem_gb"] = np.sum(allocations)
    report["total_over_alloc_gb"] = np.sum(over_alloc_mem)
    report["total_under_alloc_gb"] = np.sum(under_alloc_mem[is_under])

    return report


def plot_allocation_comparison(all_stats: list[dict], output_path: Path):
    """
    Generates a single stacked bar chart comparing the allocation performance
    of the baseline against all evaluated models.
    """
    plot_data = []
    for stats in all_stats:
        row = {'Model': stats['model_name']}
        for label in LABELS:
            row[label] = stats.get(f"{label}_perc", 0)
        plot_data.append(row)

    df = pd.DataFrame(plot_data).set_index('Model')
    df = df[LABELS]

    fig, ax = plt.subplots(figsize=(16, 10))
    df.plot(kind='bar', stacked=True, color=[
            COLORS[label] for label in LABELS], ax=ax, width=0.8)

    for c in ax.containers:
        labels = [f'{w:.1f}%' if w > 1 else '' for w in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center',
                     fontsize=12, fontweight='bold', color='black')

    ax.set_title("Memory Allocation Comparison",
                 fontsize=28, fontweight='bold', pad=20)
    ax.set_ylabel("Share of Jobs (%)", fontsize=18,
                  fontweight='bold', labelpad=15)
    ax.set_xlabel("Allocation Strategy", fontsize=18,
                  fontweight='bold', labelpad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=14, labelright=False)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Allocation Category', fontsize=12,
              title_fontsize=14, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Combined allocation comparison plot saved to {output_path}")


def generate_summary_report(all_results: list[dict], output_path: Path):
    """
    Generates a single CSV file summarizing the allocation performance
    of the baseline and all evaluated models.
    """
    df = pd.DataFrame(all_results)

    # Try to merge regression/classification results for metrics
    import os
    from .config import Config
    reg_path = Config.REGRESSION_RESULTS_CSV_PATH
    class_path = Config.CLASSIFICATION_RESULTS_CSV_PATH
    reg_df = None
    class_df = None
    if os.path.exists(reg_path):
        reg_df = pd.read_csv(reg_path)
        # Rename 'model' to 'model_name' for join
        if 'model' in reg_df.columns:
            reg_df = reg_df.rename(columns={'model': 'model_name'})
    if os.path.exists(class_path):
        class_df = pd.read_csv(class_path)
        if 'model' in class_df.columns:
            class_df = class_df.rename(columns={'model': 'model_name'})

    # Merge metrics into allocation summary
    metrics_cols = ['score_cv', 'score_hold', 'avg_pred_time']
    for metrics_df in [reg_df, class_df]:
        if metrics_df is not None:
            # Only bring in relevant columns
            merge_cols = ['model_name'] + [c for c in metrics_cols if c in metrics_df.columns]
            df = df.merge(metrics_df[merge_cols], on='model_name', how='left', suffixes=('', '_metrics'))

    # If metrics from the second merge landed in *_metrics columns, coalesce them
    for base_col in metrics_cols:
        metrics_col = f"{base_col}_metrics"
        if base_col not in df.columns and metrics_col in df.columns:
            df[base_col] = df[metrics_col]
        elif base_col in df.columns and metrics_col in df.columns:
            df[base_col] = df[base_col].fillna(df[metrics_col])
        # Drop the helper column if present
        if metrics_col in df.columns:
            df = df.drop(columns=[metrics_col])

    id_col = ['model_name']
    timing_col = ['avg_pred_time'] if 'avg_pred_time' in df.columns else []
    score_col = ['score_hold'] if 'score_hold' in df.columns else []
    cv_col = ['score_cv'] if 'score_cv' in df.columns else []
    count_cols = sorted([c for c in df.columns if c.endswith('_jobs')])
    perc_cols = sorted([c for c in df.columns if c.endswith('_perc')])
    mem_cols = sorted([c for c in df.columns if c.endswith('_gb')])

    df = df[id_col + timing_col + score_col + cv_col + count_cols + perc_cols + mem_cols]

    for col in df.columns:
        if '_perc' in col or '_gb' in col:
            df[col] = df[col].map('{:.2f}'.format)

    df.to_csv(output_path, index=False)
    print(f"Unified allocation summary report saved to {output_path}")

    # Print summary of results including prediction times and holdout scores if available
    print("\nModel Performance Summary:")

    # Filter out baseline entries for the table display
    model_rows = df[df['model_name'] != 'Baseline']

    if model_rows.empty:
        print("No model results to display.")
        return

    # Calculate max widths for alignment
    max_name_width = max(len(row['model_name']) for _, row in model_rows.iterrows())

    # Print header
    header = f"{'Model':<{max_name_width}} | {'CV Score':<10} | {'Holdout Score':<14} | {'Avg Pred Time (s)':<18}"
    print(header)
    print("-" * len(header))

    for _, row in model_rows.iterrows():
        model_name = row['model_name']

        # Format CV score
        if 'score_cv' in row and not pd.isna(row['score_cv']):
            cv_score = f"{float(row['score_cv']):.4f}"
        else:
            cv_score = "N/A"

        # Format holdout score
        if 'score_hold' in row and not pd.isna(row['score_hold']):
            holdout_score = f"{float(row['score_hold']):.4f}"
        else:
            holdout_score = "N/A"

        # Format prediction time
        if 'avg_pred_time' in row and not pd.isna(row['avg_pred_time']):
            avg_time = f"{float(row['avg_pred_time']):.6f}"
        else:
            avg_time = "N/A"

        # Print aligned row
        print(f"{model_name:<{max_name_width}} | {cv_score:<10} | {holdout_score:<14} | {avg_time:<18}")
