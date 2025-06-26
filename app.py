import streamlit as st
import pandas as pd
import joblib
import time
from data_preprocessing import DataPreprocessor

# --- Load models ---
rf_model = joblib.load("results/models/random_forest_model_partial_dataset.pkl")
xgb_model = joblib.load("results/models/xgboost_model_partial_dataset.pkl")

# --- Load and preprocess data ---
file_path = "sap_data/split_data/simulate_data.csv"
simulate_df = pd.read_csv(file_path, delimiter=";")

preprocessor = DataPreprocessor()
_, _, feature_columns, preprocessed_df = preprocessor.preprocess_pipeline(file_path)
preprocessed_df = preprocessed_df.sort_values("time").reset_index(drop=True)

# --- Streamlit App ---
st.title("📊 Real-time Memory Usage Prediction Simulator")

# Sidebar: user sets number of time steps and delay
num_steps = st.sidebar.slider("Total number of steps to simulate", 1, min(100, len(preprocessed_df)), 10)
delay_seconds = st.sidebar.slider("Delay between steps (seconds)", 1, 5, 2)

st.write("### Simulation Progress")
output_placeholder = st.empty()
chart_placeholder = st.empty()

# --- Initialize results for plotting ---
results = {
    "time": [],
    "RF Prediction": [],
    "XGB Prediction": [],
    "Actual": []
}

# --- Streaming loop ---
for i in range(num_steps):
    row = preprocessed_df.iloc[i]
    current_time = row["time"]
    input_features = pd.DataFrame([row[feature_columns]])

    # Run predictions
    rf_pred = rf_model.predict(input_features)[0]
    xgb_pred = xgb_model.predict(input_features)[0]
    actual = row["max_rss_mb"]

    # Save results
    results["time"].append(current_time)
    results["RF Prediction"].append(rf_pred)
    results["XGB Prediction"].append(xgb_pred)
    results["Actual"].append(actual)

    # Show latest predictions
    output_placeholder.write(f"[{current_time}]  \n"
                             f"🟦 RF: **{rf_pred:.2f} MB**  \n"
                             f"🟥 XGB: **{xgb_pred:.2f} MB**  \n"
                             f"🎯 Actual: **{actual:.2f} MB**  \n"
                             f"🧮 RF Error: {(rf_pred - actual):+.2f} MB | XGB Error: {(xgb_pred - actual):+.2f} MB")

    # Update chart
    plot_df = pd.DataFrame(results)
    chart_placeholder.line_chart(plot_df.set_index("time")[["RF Prediction", "XGB Prediction", "Actual"]])

    # Wait before next step
    time.sleep(delay_seconds)
