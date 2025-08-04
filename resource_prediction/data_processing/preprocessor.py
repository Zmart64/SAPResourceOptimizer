import pandas as pd
import numpy as np

# Import the configuration class to access all paths and parameters
from resource_prediction.config import Config


class DataPreprocessor:
    """
    Handles loading raw data, performing all feature engineering, splitting the data
    into chronological train and test sets, and saving the artifacts to disk.
    """

    def __init__(self, config: Config):
        """
        Initializes the preprocessor with the project configuration.

        Args:
            config (Config): The project's configuration object.
        """
        self.config = config

    def _split_build_profile(self, profile_string: str) -> pd.Series:
        """
        Helper function to split the 'buildProfile' string into its components.

        Args:
            profile_string (str): The input string, e.g., "x86_64-gcc-O2".

        Returns:
            pd.Series: A pandas Series with architecture, compiler, and optimization.
        """
        if not isinstance(profile_string, str):
            return pd.Series(["unknown"] * 3, index=["bp_arch", "bp_compiler", "bp_opt"])

        parts = profile_string.split('-')
        return pd.Series([
            parts[0],
            parts[1] if len(parts) > 1 else "unknown",
            parts[2] if len(parts) > 2 else "unknown"
        ], index=["bp_arch", "bp_compiler", "bp_opt"])

    def process(self):
        """
        Executes the full preprocessing pipeline:
        1. Loads raw data from the path specified in the config.
        2. Performs extensive feature engineering (datetime, lags, etc.).
        3. Splits the data chronologically into training and test sets.
        4. Saves the four resulting dataframes (X_train, y_train, X_test, y_test)
           as pickle files for later use.
        """
        print("--- Starting Preprocessing and Data Splitting ---")

        print("--- 1. Loading and Preparing Raw Data ---")
        df = pd.read_csv(self.config.RAW_DATA_PATH, sep=";")

        # Ensure chronological data
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df = df.sort_values('time').reset_index(drop=True)

        # Process target variable
        df[self.config.TARGET_COLUMN_RAW] = pd.to_numeric(
            df[self.config.TARGET_COLUMN_RAW], errors='coerce'
        )
        df.dropna(subset=[self.config.TARGET_COLUMN_RAW], inplace=True)
        df[self.config.TARGET_COLUMN_PROCESSED] = df[self.config.TARGET_COLUMN_RAW] / \
            (1024**3)

        print("--- 2. Feature Engineering ---")
        F = df.copy()

        # Datetime features
        F["ts_year"] = F["time"].dt.year
        F["ts_month"] = F["time"].dt.month
        F["ts_dow"] = F["time"].dt.dayofweek
        F["ts_hour"] = F["time"].dt.hour
        F["ts_dayofyear"] = F["time"].dt.dayofyear

        # Categorical features from strings
        F[["bp_arch", "bp_compiler", "bp_opt"]] = F["buildProfile"].apply(
            self._split_build_profile)
        F["target_cnt"] = F["targets"].astype(str).str.count(",") + 1

        # Lag and rolling features (time-series specific)
        lag_group_cols = ["component", "bp_arch",
                          "bp_compiler", "bp_opt", "makeType"]
        for col in lag_group_cols:
            if col in F.columns:
                F[col] = F[col].fillna('unknown_in_group_key')

        F["lag_max_rss_g1_w1"] = F.groupby(lag_group_cols, observed=True)["max_rss"].transform(
            lambda s: s.shift(1).rolling(window=1, min_periods=1).mean()
        )
        F['rolling_p95_rss_g1_w5'] = F.groupby(lag_group_cols, observed=True)["max_rss"].transform(
            lambda s: s.shift(1).rolling(
                window=5, min_periods=1).quantile(0.95)
        )

        # Final set of features
        features_to_use = [
            "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_dayofyear", "bp_arch",
            "bp_compiler", "bp_opt", "target_cnt", "lag_max_rss_g1_w1",
            "rolling_p95_rss_g1_w5", "component", "makeType", "jobs", "localJobs"
        ]

        X = F[features_to_use].copy()
        y = F[[self.config.TARGET_COLUMN_PROCESSED,
               self.config.TARGET_COLUMN_RAW]].copy()

        # Cleaning and type conversion
        cat_features = X.select_dtypes(
            include=['object', 'category']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype('category')

        for col in X.select_dtypes(include=np.number).columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        print(
            f"--- 3. Splitting data with test set fraction: {self.config.TEST_SET_FRACTION} ---")

        # Chronological split
        split_index = int(len(X) * (1 - self.config.TEST_SET_FRACTION))

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        print(f"Total samples: {len(X)}")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples (Holdout set)")

        print("--- 4. Saving processed data to disk ---")
        self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        X_train.to_pickle(self.config.X_TRAIN_PATH)
        y_train.to_pickle(self.config.Y_TRAIN_PATH)
        X_test.to_pickle(self.config.X_TEST_PATH)
        y_test.to_pickle(self.config.Y_TEST_PATH)

        print(
            f"Train/Test data successfully saved to '{self.config.PROCESSED_DATA_DIR.resolve()}'")

        print("\n--- Preprocessing and Splitting Complete ---")
