"""
Core Pareto utilities: generate frontier by retraining for alphas, load results,
compute Pareto frontier, and pick key points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from resource_prediction.config import Config
from resource_prediction.models.implementations.quantile_ensemble_variants import (
    LGBXGBQuantileEnsemble,
)


def _allocation_metrics(allocated: np.ndarray, true: np.ndarray) -> dict:
    """Compute allocation metrics consistent with training/evaluation.

    - under_pct: percentage of jobs where allocated < true
    - total_over_pct: 100 * sum(max(0, alloc-true)) / sum(true)
    - business_score: 5*under_pct + total_over_pct
    """
    true = np.asarray(true)
    allocated = np.asarray(allocated)
    n = len(true)
    under = np.sum(allocated < true)
    over = np.maximum(0, allocated - true)
    under_pct = 100 * under / n if n > 0 else 0.0
    denom = true.sum()
    total_over_pct = 100 * over.sum() / denom if denom > 0 else 0.0
    business_score = 5 * under_pct + total_over_pct
    return {
        "under_pct": under_pct,
        "total_over_pct": total_over_pct,
        "business_score": business_score,
    }


def _is_dominated(p: pd.Series, others: pd.DataFrame) -> bool:
    """Return True if point p is dominated by any row in others (minimization)."""
    return any(
        (o["under_pct"] <= p["under_pct"]) and (o["total_over_pct"] <= p["total_over_pct"]) and (
            (o["under_pct"] < p["under_pct"]) or (o["total_over_pct"] < p["total_over_pct"])  # at least one strict
        )
        for _, o in others.iterrows()
    )


def _to_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to non-dominated (Pareto-optimal) points."""
    mask = []
    for i, row in df.iterrows():
        dominated = _is_dominated(row, df.drop(index=i))
        mask.append(not dominated)
    return df.loc[mask].reset_index(drop=True)


def _get_best_lgb_xgb_hparams(cfg: Config) -> dict | None:
    """Read champion CSV and extract best hyperparameters for lgb_xgb_ensemble.

    Falls back to defaults if file is missing or row not found.
    """
    try:
        df = pd.read_csv(cfg.REGRESSION_RESULTS_CSV_PATH)
        df = df[df["model"] == "lgb_xgb_ensemble"].sort_values("score_cv")
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "lgb_n_estimators": int(row.get("lgb_n_estimators", 300)),
            "lgb_num_leaves": int(row.get("lgb_num_leaves", 31)),
            "lgb_lr": float(row.get("lgb_lr", 0.05)),
            "xgb_n_estimators": int(row.get("xgb_n_estimators", 300)),
            "xgb_max_depth": int(row.get("xgb_max_depth", 6)),
            "xgb_lr": float(row.get("xgb_lr", 0.05)),
        }
    except Exception:
        return None


def generate_frontier(cfg: Config, alphas: Iterable[float] | None = None, safeties: Iterable[float] | None = None,
                      save_all_points: bool = True) -> Path:
    """Train LGB+XGB QE for each alpha and evaluate across safety grid to build a frontier CSV.

    Uses the champion hyperparameters for the lgb_xgb_ensemble (if available) to keep points comparable.

    Returns the path to the saved CSV of Pareto points.
    """
    if alphas is None:
        alphas = [0.90, 0.95, 0.98, 0.99]
    if safeties is None:
        safeties = np.round(np.linspace(1.00, 1.15, 8), 3)

    # Load processed splits
    X_train = pd.read_pickle(cfg.X_TRAIN_PATH)
    y_train = pd.read_pickle(cfg.Y_TRAIN_PATH)[cfg.TARGET_COLUMN_PROCESSED]
    X_test = pd.read_pickle(cfg.X_TEST_PATH)
    y_test = pd.read_pickle(cfg.Y_TEST_PATH)[cfg.TARGET_COLUMN_PROCESSED]

    # Determine features and hyperparameters
    best = _get_best_lgb_xgb_hparams(cfg) or {
        "lgb_n_estimators": 300,
        "lgb_num_leaves": 31,
        "lgb_lr": 0.05,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 6,
        "xgb_lr": 0.05,
    }
    features = cfg.BASE_FEATURES
    X_train_fs = X_train[features]
    X_test_fs = X_test[features]

    rows = []
    # For each alpha, retrain the base learners; then sweep safety post-hoc on predictions
    for alpha in alphas:
        predictor = LGBXGBQuantileEnsemble(
            alpha=alpha,
            safety=1.0,  # set to 1.0 for raw predictions; apply safety downstream
            lgb_n_estimators=best["lgb_n_estimators"],
            lgb_lr=best["lgb_lr"],
            lgb_max_depth=None,  # respect LightGBM defaults; depth can be None
            xgb_n_estimators=best["xgb_n_estimators"],
            xgb_max_depth=best["xgb_max_depth"],
            xgb_lr=best["xgb_lr"],
            random_state=cfg.RANDOM_STATE,
        )
        predictor.fit(X_train_fs, y_train)
        raw = predictor.predict(X_test_fs)  # currently safety=1.0

        for s in safeties:
            alloc = raw * s
            mets = _allocation_metrics(alloc, y_test.values)
            rows.append({
                "alpha": alpha,
                "safety": float(s),
                **mets,
            })

    points = pd.DataFrame(rows)
    # Compute Pareto frontier
    frontier = _to_frontier(points)

    results_dir = cfg.PROJECT_ROOT / "artifacts" / "pareto" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_path = results_dir / "pareto_all_points.csv"
    front_path = results_dir / "pareto_frontier_points.csv"
    if save_all_points:
        points.to_csv(all_path, index=False)
    frontier.to_csv(front_path, index=False)
    return front_path


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
