"""Data preprocessing pipeline for build telemetry."""

import pandas as pd
import numpy as np
from resource_prediction.config import Config


class DataPreprocessor:
    """
    Handles loading, comprehensive feature engineering, and data splitting.

    This class serves as the unified preprocessor for both regression and
    classification tasks, ensuring consistent data handling across the project.
    """

    def __init__(self, config: Config):
        """
        Initializes the preprocessor with the project configuration.

        Args:
            config (Config): The project's configuration object.
        """
        self.config = config

    def _split_build_profile(self, s: str) -> list:
        """Splits the 'buildProfile' string into its three components."""
        if not isinstance(s, str):
            return ["unknown"] * 3
        parts = (s.split("-") + ["unknown"] * 3)[:3]
        return parts

    def process(self):
        """
        Executes the full preprocessing pipeline: loading, feature engineering,
        splitting, and saving data artifacts.
        """
        print("Starting unified preprocessing and data splitting...")

        print("Step 1: Loading and preparing raw data")
        df = pd.read_csv(self.config.RAW_DATA_PATH, sep=";")
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df = df.sort_values('time').reset_index(drop=True)
        df[self.config.TARGET_COLUMN_RAW] = pd.to_numeric(
            df[self.config.TARGET_COLUMN_RAW], errors='coerce')
        df.dropna(subset=[self.config.TARGET_COLUMN_RAW], inplace=True)
        df[self.config.TARGET_COLUMN_PROCESSED] = df[self.config.TARGET_COLUMN_RAW] / \
            (1024**3)

        print("Step 2: Performing comprehensive feature engineering")
        F = df.copy()

        F["ts_year"] = F["time"].dt.year
        F["ts_month"] = F["time"].dt.month
        F["ts_dow"] = F["time"].dt.dayofweek
        F["ts_hour"] = F["time"].dt.hour
        F["ts_dayofyear"] = F["time"].dt.dayofyear
        F["ts_weekofyear"] = F["time"].dt.isocalendar().week

        F[["bp_arch", "bp_compiler", "bp_opt"]] = F["buildProfile"].apply(
            self._split_build_profile).tolist()
        F["branch_id_str"] = pd.to_numeric(F["branch"].str.extract(
            r"(\d+)$")[0], errors="coerce").fillna(-1)
        F["branch_prefix"] = F["branch"].str.replace(
            r"[\d_]*$", "", regex=True).replace('', 'unknown_prefix')

        F["target_cnt"] = F["targets"].astype(str).str.count(",") + 1
        F["target_has_dist"] = F["targets"].astype(
            str).str.contains("dist").astype(int)
        F["build_load"] = F["jobs"] + F["localJobs"]
        F["target_intensity"] = F["targets"].astype(str).str.len() / 100.0
        F["debug_multiplier"] = F["bp_opt"].str.contains(
            "debug|asan|tsan", case=False).astype(int)
        F["heavy_target_flag"] = F["targets"].str.contains(
            "all|dist|install", case=False).astype(int)
        F["high_parallelism"] = (F["localJobs"] > 8).astype(int)

        grp_cols = ["component", "bp_arch",
                    "bp_compiler", "bp_opt", "makeType"]
        for c in grp_cols:
            if c in F.columns:
                F[c] = F[c].fillna("unknown")

        F['lag_1_grouped'] = F.groupby(
            grp_cols)["max_rss_gb"].transform('shift', 1)
        F['rolling_p95_rss_g1_w5'] = F.groupby(grp_cols, observed=True)[
            "max_rss_gb"].transform(lambda s: s.shift(1).rolling(5, 1).quantile(0.95))
        F["lag_max_rss_global_w5"] = F["max_rss_gb"].shift(
            1).rolling(5, 1).mean()

        for col in F.select_dtypes("number").columns:
            if F[col].isnull().any():
                F[col] = F[col].fillna(F[col].median())

        X = F[self.config.ALL_FEATURES].copy()
        y = F[[self.config.TARGET_COLUMN_PROCESSED,
               self.config.TARGET_COLUMN_RAW]].copy()

        print(
            f"Step 3: Splitting data with test set fraction: {self.config.TEST_SET_FRACTION}")
        split_index = int(len(X) * (1 - self.config.TEST_SET_FRACTION))
        X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
        X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

        self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_pickle(self.config.X_TRAIN_PATH)
        y_train.to_pickle(self.config.Y_TRAIN_PATH)
        X_test.to_pickle(self.config.X_TEST_PATH)
        y_test.to_pickle(self.config.Y_TEST_PATH)
        print(
            f"Train/Test data successfully saved to '{self.config.PROCESSED_DATA_DIR.resolve()}'")
        print("Preprocessing and splitting complete.")
