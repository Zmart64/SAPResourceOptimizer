import joblib
import pandas as pd
import numpy as np

# This is a placeholder for your actual preprocessing logic.
# from data_preprocessing import DataPreprocessor

# --- Configuration ---
# Path to the saved model payload from your training script
MODEL_PAYLOAD_PATH = "output_model_and_plots/xgb_classifier_model.pkl"
# Path to the new data you want to predict on
FILE_PATH = "build-data-sorted.csv"  # Using the training data for demonstration

# --- 1. Load Model and All Necessary Metadata ---
print(f"Loading model payload from: {MODEL_PAYLOAD_PATH}")
try:
    # The saved file is a dictionary containing the model, bin edges, and feature list
    model_payload = joblib.load(MODEL_PAYLOAD_PATH)
except FileNotFoundError:
    print(f"ERROR: Model file not found at '{MODEL_PAYLOAD_PATH}'.")
    print("Please run the training script first to generate the model file.")
    exit()

# Extract all the necessary components from the payload
xgb_classifier = model_payload['model']
feature_columns_from_model = model_payload['feature_columns']
bin_edges_gb = model_payload['bin_edges_gb']

print("Model and metadata loaded successfully.")
print(f"Model expects {len(feature_columns_from_model)} features.")
print(f"Memory Bins (GB): {np.round(bin_edges_gb, 2)}")


# --- 2. Load and Preprocess the Simulation Data ---
# In your real use case, you would use your DataPreprocessor here.
# For this example, we perform minimal preprocessing to create the necessary columns.
print(f"\nLoading and preprocessing simulation data from: {FILE_PATH}")
simulate_df = pd.read_csv(FILE_PATH, delimiter=";")

# --- Minimal Preprocessing for Demonstration ---
# This must create all the columns listed in `feature_columns_from_model`
# and a `max_rss_gb` column for comparison.


def _split_build_profile(profile_string):
    if not isinstance(profile_string, str):
        return pd.Series(["unknown"]*3)
    parts = profile_string.split('-')
    return pd.Series([parts[0], parts[1] if len(parts) > 1 else "unknown", parts[2] if len(parts) > 2 else "unknown"])


simulate_df['time'] = pd.to_datetime(simulate_df['time'])
simulate_df = simulate_df.sort_values("time").reset_index(drop=True)
simulate_df['max_rss_gb'] = simulate_df['max_rss'] / \
    (1024**3)  # Target for comparison
simulate_df['target_cnt'] = simulate_df['targets'].astype(
    str).str.count(",") + 1
simulate_df[["bp_arch", "bp_compiler", "bp_opt"]
            ] = simulate_df["buildProfile"].apply(_split_build_profile)
simulate_df["branch_prefix"] = simulate_df["branch"].str.replace(
    r"[\d_]*$", "", regex=True).replace('', 'unknown_prefix')
simulate_df["ts_month"] = simulate_df["time"].dt.month
simulate_df["ts_dow"] = simulate_df["time"].dt.dayofweek
simulate_df["ts_hour"] = simulate_df["time"].dt.hour

# Convert object columns to 'category' dtype as expected by the model
for col in simulate_df.select_dtypes(include=['object']).columns:
    if col in feature_columns_from_model:
        simulate_df[col] = simulate_df[col].astype('category')
# --- End of Preprocessing ---


# --- 3. Simulate Real-Time Row-by-Row Prediction ---
print("\n--- Starting Real-Time Simulation (predicting last 15 data points) ---\n")

# Use a small slice of the data for the simulation
preprocessed_df = simulate_df.tail(15).reset_index(drop=True)

for i, row in preprocessed_df.iterrows():
    current_time = row['time']

    # --- Prepare input for model ---
    # Create a single-row DataFrame with the exact columns the model was trained on
    input_df = pd.DataFrame([row])[feature_columns_from_model]

    # Ensure dtypes are correct (especially for categoricals)
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype('category')

    # --- Predict the CLASS ---
    # The model outputs a class index (e.g., 0, 1, 2...)
    predicted_class = xgb_classifier.predict(input_df)[0]

    # --- Translate the CLASS into a MEMORY ALLOCATION ---
    # Strategy: Allocate the upper bound of the predicted class's bin
    allocated_mem_gb = bin_edges_gb[min(
        predicted_class + 1, len(bin_edges_gb) - 1)]

    # Get the actual value for comparison (must be in the same units, GB)
    actual_gb = row['max_rss_gb']

    # Calculate the difference
    difference_gb = allocated_mem_gb - actual_gb
    status = "OVER" if difference_gb > 0 else "UNDER" if difference_gb < 0 else "PERFECT"

    print(
        f"[{current_time}] "
        f"Class Pred: {predicted_class} -> "
        f"Allocated: {allocated_mem_gb:6.2f} GB | "
        f"Actual: {actual_gb:6.2f} GB | "
        f"Difference: {difference_gb:6.2f} GB ({status})"
    )

print("\nSimulation complete.")
