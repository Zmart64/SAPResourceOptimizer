"""
Core Pareto functions: load results, compute Pareto frontier, pick key points.
"""
import pandas as pd


def load_results(results_csv: str) -> pd.DataFrame:
    """Load the Pareto search results from CSV."""
    return pd.read_csv(results_csv)


def load_frontier(points_csv: str) -> pd.DataFrame:
    """Load only the Pareto-optimal points."""
    return pd.read_csv(points_csv)


def get_key_points(pareto_df: pd.DataFrame) -> dict:
    """Identify low-waste, low-underallocation, and balanced configurations."""
    idx = {
        'low_waste': pareto_df['total_over_pct'].idxmin(),
        'low_underallocation': pareto_df['under_pct'].idxmin(),
        'balanced': pareto_df['business_score'].idxmin()
    }
    return {k: pareto_df.loc[i] for k, i in idx.items()}
