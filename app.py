import streamlit as st
import altair as alt
import pandas as pd
import joblib
import time
from collections import deque
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

# Sidebar: Delay control
delay_seconds = st.sidebar.slider("Delay between steps (seconds)", 1, 5, 2)

st.write("### Streaming Predictions (Last 10 steps)")
output_placeholder = st.empty()
chart_placeholder = st.empty()

# --- Initialize deque (sliding window of size 10) ---
window_size = 10
results = {
    "time": deque(maxlen=window_size),
    "RF Prediction": deque(maxlen=window_size),
    "XGB Prediction": deque(maxlen=window_size),
    "Actual": deque(maxlen=window_size)
}

# --- Streaming loop ---
i = 0
while i < len(preprocessed_df):
    row = preprocessed_df.iloc[i]
    current_time = row["time"]
    input_features = pd.DataFrame([row[feature_columns]])

    # Run predictions
    rf_pred = rf_model.predict(input_features)[0]
    xgb_pred = xgb_model.predict(input_features)[0]
    actual = row["max_rss_mb"]

    # Update results window
    results["time"].append(current_time)
    results["RF Prediction"].append(rf_pred)
    results["XGB Prediction"].append(xgb_pred)
    results["Actual"].append(actual)

    # Calculate errors
    rf_error = rf_pred - actual
    xgb_error = xgb_pred - actual

    # Color classification using Streamlit's new markdown
    def colorize_error(val):
        if val < 0:
            color = "red"
        elif val <= 20000:
            color = "green"
        else:
            color = "orange"
        return f":{color}[{val:+.2f} MB]"

    # Display latest prediction
    output_placeholder.markdown(f"""
                                **[{current_time}]**  
                                🟨 RF: `{rf_pred:.2f} MB`  
                                🟦 XGB: `{xgb_pred:.2f} MB`  
                                🟥 Actual: `{actual:.2f} MB`  
                                RF Error: {colorize_error(rf_error)} | XGB Error: {colorize_error(xgb_error)}
                                """)

    # Update chart with only last 10 points
    plot_df = pd.DataFrame(results)
    # Melt the dataframe for Altair (long format)
    melted_df = plot_df.melt(id_vars="time", var_name="Model", value_name="Memory")

    # Define color mapping
    color_scale = alt.Scale(
    domain=["RF Prediction", "XGB Prediction", "Actual"],
    range=["#d0b51c", "#1da4b3", "red"]  
    )

    # Create line chart
    line_chart = alt.Chart(melted_df).mark_line(point=True).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("Memory:Q", title="Memory (MB)"),
        color=alt.Color("Model:N", 
                        title="Model",
                        scale=alt.Scale(
                            domain=["RF Prediction", "XGB Prediction", "Actual"],
                            range=["#f8e515", "#13d6c9", "red"]
                        )),
        tooltip=["time:T", "Model:N", "Memory:Q"]
    ).properties(
        width=700,
        height=400
)

    chart_placeholder.altair_chart(line_chart, use_container_width=True)

    time.sleep(delay_seconds)
    i += 1
