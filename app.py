import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque

# --- Configuration ---
MODEL_PAYLOAD_PATH = "output_model_and_plots/xgb_classifier_model.pkl"
DATA_TO_SIMULATE_PATH = "output_model_and_plots/simulation_data.csv"

# --- 1. Load Model and All Necessary Metadata ---
st.set_page_config(layout="wide")


@st.cache_resource
def load_model_payload():
    """Loads the entire model payload (model, features, bins) from the pkl file."""
    try:
        payload = joblib.load(MODEL_PAYLOAD_PATH)
        return payload
    except FileNotFoundError:
        st.error(f"FATAL: Model file not found at '{MODEL_PAYLOAD_PATH}'.")
        st.error("Please run the training script first to generate the model file.")
        return None


model_payload = load_model_payload()
if not model_payload:
    st.stop()

# Extract all the necessary components from the payload
xgb_classifier = model_payload['model']
feature_columns_from_model = model_payload['feature_columns']
bin_edges_gb = model_payload['bin_edges_gb']


# --- 2. Load and Preprocess the Simulation Data ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses data to match the model's training features."""
    try:
        df = pd.read_csv(file_path, delimiter=";")
    except FileNotFoundError:
        st.error(f"FATAL: Simulation data file not found at '{file_path}'.")
        st.error(
            "Please ensure the training script has been run and the file was created.")
        return None

    def _split_build_profile(profile_string):
        if not isinstance(profile_string, str):
            return pd.Series(["unknown"]*3)
        parts = profile_string.split('-')
        return pd.Series([parts[0], parts[1] if len(parts) > 1 else "unknown", parts[2] if len(parts) > 2 else "unknown"])

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values("time").reset_index(drop=True)
    df['max_rss_gb'] = df['max_rss'] / (1024**3)
    df['target_cnt'] = df['targets'].astype(str).str.count(",") + 1
    df[["bp_arch", "bp_compiler", "bp_opt"]
       ] = df["buildProfile"].apply(_split_build_profile)
    df["branch_prefix"] = df["branch"].str.replace(
        r"[\d_]*$", "", regex=True).replace('', 'unknown_prefix')
    df["ts_month"] = df["time"].dt.month
    df["ts_dow"] = df["time"].dt.dayofweek
    df["ts_hour"] = df["time"].dt.hour

    for col in df.select_dtypes(include=['object']).columns:
        if col in feature_columns_from_model:
            df[col] = df[col].astype('category')
    return df


preprocessed_df = load_and_preprocess_data(DATA_TO_SIMULATE_PATH)
if preprocessed_df is None:
    st.stop()


# --- 3. Streamlit App ---
st.title("📊 Real-time Memory Classification & Allocation Simulator")

# --- Sidebar ---
st.sidebar.header("Simulation Controls")
delay_seconds = st.sidebar.slider(
    "Delay between steps (seconds)", 0.1, 5.0, 0.5)
st.sidebar.info("This app simulates real-time data arriving one row at a time. The model predicts a memory 'class', which is then translated into a GB allocation.")
st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Features Used:** `{len(feature_columns_from_model)}`")
st.sidebar.write(f"**Memory Bins (GB):**")
st.sidebar.code(np.round(bin_edges_gb, 2))

# --- NEW: Placeholders for Cumulative Summary Metrics ---
st.subheader("Overall Simulation Summary")
summary_placeholder = st.empty()


# --- Main Display Area ---
st.write("### Streaming Predictions (Live Sliding Window)")
col1, col2 = st.columns([1, 2])
output_placeholder = col1.empty()
chart_placeholder = col2.empty()

# --- Initialize deques and NEW cumulative trackers ---
window_size = 20
results = {
    "Time": deque(maxlen=window_size),
    "Allocated Memory (GB)": deque(maxlen=window_size),
    "Actual Memory (GB)": deque(maxlen=window_size)
}
# Cumulative trackers
total_over_allocated_gb = 0.0
total_under_allocated_gb = 0.0
jobs_under_allocated_count = 0
total_jobs_processed = 0


# --- Streaming loop ---
for i, row in preprocessed_df.iterrows():
    total_jobs_processed += 1
    current_time = row["time"]

    # Prepare a single-row DataFrame
    input_df = pd.DataFrame([row])[feature_columns_from_model]
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype('category')

    # --- Run Prediction, Translate, and get Actual ---
    predicted_class = xgb_classifier.predict(input_df)[0]
    allocated_mem_gb = bin_edges_gb[min(
        predicted_class + 1, len(bin_edges_gb) - 1)]
    actual_gb = row["max_rss_gb"]
    difference_gb = allocated_mem_gb - actual_gb

    # --- NEW: Update cumulative trackers ---
    if difference_gb < 0:
        # This is an under-allocation
        jobs_under_allocated_count += 1
        total_under_allocated_gb += abs(difference_gb)
    else:
        # This is an over-allocation (or perfect allocation)
        total_over_allocated_gb += difference_gb

    # --- Update the cumulative summary display ---
    with summary_placeholder.container():
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Jobs Processed", total_jobs_processed)

        # Calculate percentage of under-allocated jobs safely
        under_alloc_percent = (jobs_under_allocated_count /
                               total_jobs_processed) * 100 if total_jobs_processed > 0 else 0
        sum_col2.metric("Jobs Under-allocated",
                        f"{jobs_under_allocated_count} ({under_alloc_percent:.1f}%)")

        sum_col3.metric("Total Under-allocated Shortfall",
                        f"{total_under_allocated_gb:.2f} GB")
        sum_col4.metric("Total Over-allocated Waste",
                        f"{total_over_allocated_gb:.2f} GB")

    # --- Update sliding window results ---
    results["Time"].append(current_time)
    results["Allocated Memory (GB)"].append(allocated_mem_gb)
    results["Actual Memory (GB)"].append(actual_gb)

    # --- Display latest prediction details ---
    with output_placeholder.container():
        st.markdown(f"**Timestamp:** `{current_time}`")
        st.metric("Predicted Class", f"Class {predicted_class}")
        st.metric("Memory Allocated", f"{allocated_mem_gb:.2f} GB")
        st.metric("Actual Memory Used", f"{actual_gb:.2f} GB",
                  f"{difference_gb:.2f} GB", delta_color="inverse")
        st.caption(
            "Delta shows (Allocated - Actual). Negative is under-allocation.")

    # --- Update chart ---
    with chart_placeholder.container():
        plot_df = pd.DataFrame(results)
        melted_df = plot_df.melt(
            id_vars="Time", var_name="Series", value_name="Memory (GB)")
        line_chart = alt.Chart(melted_df).mark_line(point=alt.OverlayMarkDef(size=50)).encode(
            x=alt.X("Time:T", title="Time"),
            y=alt.Y("Memory (GB):Q", title="Memory (GB)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title="Data Series", scale=alt.Scale(domain=[
                            "Allocated Memory (GB)", "Actual Memory (GB)"], range=["#1f77b4", "#d62728"])),
            tooltip=["Time:T", "Series:N", alt.Tooltip(
                "Memory (GB):Q", format=".2f")]
        ).properties(title="Allocated vs. Actual Memory Usage", height=400).interactive()
        st.altair_chart(line_chart, use_container_width=True)

    time.sleep(delay_seconds)
