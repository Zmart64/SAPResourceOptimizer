"""
Data Preprocessing
"""

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import Config


class DataPreprocessor:
    """Data Preprocessing class for max_rss prediction"""

    def __init__(self):
        self.label_encoders = {}

    def load_data(self, file_path) -> pd.DataFrame:
        """Load the build data CSV file"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path, delimiter=";")
        print(f"Dataset shape: {df.shape}")
        return df

    def convert_memory_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert memory values from bytes to MB for better interpretability"""
        df["max_rss_mb"] = df["max_rss"] / (1024 * 1024)
        df["max_cache_mb"] = df["max_cache"] / (1024 * 1024)
        # Add memreq_mb conversion
        df["memreq_mb"] = df["memreq"]
        return df

    def filter_successful_builds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only successful builds (ts_status == 'ok')"""
        print(f"Original dataset rows: {len(df)}")
        print(f"\nStatus distribution:\n{df['ts_status'].value_counts()}")

        df_clean = df[df["ts_status"] == "ok"].copy()
        print(f"\nFiltered to successful builds: {len(df_clean)} rows")

        # Add this check
        if len(df_clean) == 0:
            print("WARNING: No successful builds found!")
        elif len(df_clean) < 100:
            print("WARNING: Very small dataset after filtering!")

        return df_clean

    def process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time column to datetime and extract time-based features"""
        df["time"] = pd.to_datetime(df["time"], format="mixed")
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        return df

    def extract_build_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from buildProfile column"""
        df["architecture"] = df["buildProfile"].str.extract(r"(linux\w+)")
        df["compiler"] = df["buildProfile"].str.extract(r"-(gcc\d*|clang\w*)")
        df["optimization"] = df["buildProfile"].str.extract(
            r"-(optimized|debug|release)"
        )

        df["architecture"] = df["architecture"].fillna("unknown")
        df["compiler"] = df["compiler"].fillna("unknown")
        df["optimization"] = df["optimization"].fillna("unknown")

        return df

    def process_targets_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process targets column to extract meaningful features"""
        # Create binary features for common target types
        df["has_all_target"] = df["targets"].str.contains("all", na=False).astype(int)
        df["has_dist_target"] = df["targets"].str.contains("dist", na=False).astype(int)
        df["has_sign_target"] = df["targets"].str.contains("sign", na=False).astype(int)
        df["target_count"] = df["targets"].str.count(",") + 1

        # Handle NaN values
        df["target_count"] = df["target_count"].fillna(1)

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prev_avg derived feature"""
        df = df.sort_values("time").reset_index(drop=True)

        # Initialize pre_avg column
        df["pre_avg"] = np.nan

        print(
            f"Creating pre_avg feature with n={Config.PRE_AVG_N} similar submissions..."
        )

        # For each row, find similar previous submissions
        for i in range(len(df)):
            current_row = df.iloc[i]
            previous_df = df.iloc[:i]

            if len(previous_df) < Config.PRE_AVG_N:
                continue

            similar_mask = pd.Series([True] * len(previous_df))

            # compare similarity features against all rows before current row
            # this is propably faster than finding just n from the last row
            # because pandas is optimzed
            for feature in Config.SIMILARITY_FEATURES:
                similar_mask &= previous_df[feature] == current_row[feature]

            similar_submissions = previous_df[similar_mask]

            # Get last n similar submissions
            if len(similar_submissions) > 0:
                last_n_similar = similar_submissions.tail(Config.PRE_AVG_N)
                df.loc[i, "pre_avg"] = last_n_similar["max_rss_mb"].mean()

        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_features
    ) -> pd.DataFrame:
        """Label encode categorical features"""
        for col in categorical_features:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list]:
        """Prepare final feature matrix for machine learning"""
        # Define feature columns
        feature_columns = [
            "jobs",
            "localJobs",
            "memreq_mb",
            "hour",
            "day_of_week",
            "pre_avg",
        ]

        # Categorical features to encode
        categorical_features = [
            "location",
            "branch",
            "makeType",
            "targets",
            "component",
            "ts_phase",
            "architecture",
            "compiler",
            "optimization",
        ]

        # Encode categorical features
        df = self.encode_categorical_features(df, categorical_features)

        # Add encoded features to feature list
        for col in categorical_features:
            feature_columns.append(f"{col}_encoded")

        # Prepare final matrices
        x = df[feature_columns].copy()
        y = df["max_rss_mb"].copy()

        return x, y, feature_columns

    def preprocess_pipeline(self, file_path: str):
        """Complete preprocessing pipeline"""
        print("=== Starting Data Preprocessing Pipeline ===")

        # Load data
        df = self.load_data(file_path)

        # Convert memory units
        df = self.convert_memory_units(df)

        # Filter successful builds
        df = self.filter_successful_builds(df)

        # Process datetime
        df = self.process_datetime(df)

        # Extract build profile features
        df = self.extract_build_profile_features(df)

        # Process targets feature
        # df = self.process_targets_feature(df)

        # Create derived features
        start_time = time.time()
        df = self.create_derived_features(df)
        end_time = time.time()
        print(f"create_derived_features took {end_time - start_time:.2f} seconds")

        # Prepare features
        x, y, feature_columns = self.prepare_features(df)

        print(f"Final feature matrix shape: {x.shape}")
        print(f"Target vector shape: {y.shape}")

        df.to_csv(f"{Config.RESULTS_DIR}/preprocessed_data.csv", index=False)
        print(f"Preprocessed data saved to: {Config.RESULTS_DIR}/preprocessed_data.csv")

        print("=== Preprocessing Complete ===")

        return x, y, feature_columns, df
