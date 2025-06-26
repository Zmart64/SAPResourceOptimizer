import joblib
import pandas as pd
import time
from data_preprocessing import DataPreprocessor

rf_model = joblib.load("results/models/random_forest_model_partial_dataset.pkl")
xgb_model = joblib.load("results/models/xgboost_model_partial_dataset.pkl")

file_path = "sap_data/split_data/simulate_data.csv"

simulate_df = pd.read_csv(file_path, delimiter=";")
preprocessor = DataPreprocessor()

_, _, feature_columns, preprocessed_df = preprocessor.preprocess_pipeline(file_path)

# Ensure the data is sorted by time
preprocessed_df = preprocessed_df.sort_values("time").reset_index(drop=True)

# Simulate real-time row-by-row prediction
window_size = 5  # for computing rolling pre_avg if needed
history = []

print("Starting simulation...\n")

for i, row in preprocessed_df.iterrows():
    if (i==10):
        break
    current_time = row['time']

    # --- Prepare input for model ---
    input_features = pd.DataFrame([row[feature_columns]])

    # --- Predict ---
    rf_pred = rf_model.predict(input_features)[0]
    xgb_pred = xgb_model.predict(input_features)[0]
    actual = row['max_rss_mb']

    print(f"[{current_time}] RF: {rf_pred:.2f} MB | XGB: {xgb_pred:.2f} MB | Actual: {actual:.2f} MB | OverPrediction RF: {(rf_pred - actual):.2f} MB | OverPrediction XGB: {(xgb_pred - actual):.2f} MB")

    # (Optional) maintain sliding window for dynamic pre_avg updates
    history.append(row)
    if len(history) > window_size:
        history.pop(0)