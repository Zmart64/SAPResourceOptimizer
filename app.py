import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque

# --- Configuration ---
# MODIFIED: Point to the model payload .pkl file
MODEL_PAYLOAD_PATH = "output_plots_best_performer/random_forest_final_tuned/final_model.pkl"
# MODIFIED: Point to the holdout data created by the training script
SIMULATION_DATA_PATH = "output_plots_best_performer/simulation_data.csv"

# --- 1. Load Model and All Necessary Metadata ---
st.set_page_config(layout="wide")


@st.cache_resource
def load_model_payload():
    """Loads the entire model payload (model, features, bins, etc.) from the pkl file."""
    try:
        # This now correctly loads the dictionary we saved
        payload = joblib.load(MODEL_PAYLOAD_PATH)
        return payload
    except FileNotFoundError:
        st.error(
            f"FATAL: Model payload file not found at '{MODEL_PAYLOAD_PATH}'.")
        st.error(
            "Please run the `train_best_performer.py` script first to generate the model file and its payload.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model payload: {e}")
        return None


model_payload = load_model_payload()
if not model_payload:
    st.stop()

# --- CORRECTED: Extract all components from the rich payload ---
# This will now work because model_payload is a dictionary
model = model_payload['model']
model_name = model_payload['model_name']
bin_edges_gb = model_payload['bin_edges_gb']
feature_columns_from_model = model_payload['feature_columns']
# These are no longer needed for processing but good for display
requires_one_hot = model_payload['requires_one_hot']
original_categorical_features = model_payload['original_categorical_features']


# --- 2. Load the PRE-PROCESSED Simulation Data ---
@st.cache_data
def load_simulation_data(file_path):
    """Loads the pre-processed simulation data. No feature engineering needed here."""
    try:
        # The holdout file already has all the features computed
        df = pd.read_csv(file_path, delimiter=";")
    except FileNotFoundError:
        st.error(f"FATAL: Simulation data file not found at '{file_path}'.")
        st.error(
            "Please ensure the training script has been run and the holdout file was created.")
        return None

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values("time").reset_index(drop=True)

    # For models that don't use one-hot encoding, we must ensure categorical
    # columns are of the 'category' dtype as they were during training.
    if not requires_one_hot:
        for col in df.select_dtypes(include=['object']).columns:
            if col in feature_columns_from_model:
                df[col] = df[col].astype('category')

    return df


# This dataframe now contains all the necessary feature columns
simulation_df = load_simulation_data(SIMULATION_DATA_PATH)
if simulation_df is None:
    st.stop()


# --- 3. Streamlit App ---
st.title(f"📊 Real-time Allocation Simulator ({model_name})")

# --- Sidebar ---
st.sidebar.header("Simulation Controls")
delay_seconds = st.sidebar.slider(
    "Delay between steps (seconds)", 0.0, 2.0, 0.2)
st.sidebar.info("This app simulates real-time data arriving one row at a time. The loaded model predicts a memory 'class', which is then translated into a GB allocation.")
st.sidebar.subheader("Loaded Model Info")
st.sidebar.markdown(f"**Model Type:** `{model_name}`")
st.sidebar.markdown(f"**Requires One-Hot Encoding:** `{requires_one_hot}`")
st.sidebar.write(f"**Memory Bins (GB):**")
st.sidebar.code(np.round(bin_edges_gb, 2))

# --- Placeholders for UI elements ---
st.subheader("Overall Simulation Summary")
summary_placeholder = st.empty()
st.write("### Streaming Predictions (Live Sliding Window)")
col1, col2 = st.columns([1, 2])
output_placeholder = col1.empty()
chart_placeholder = col2.empty()

# --- Initialize deques and cumulative trackers ---
window_size = 20
results = {"Time": deque(maxlen=window_size), "Allocated Memory (GB)": deque(
    maxlen=window_size), "Actual Memory (GB)": deque(maxlen=window_size)}
total_over_allocated_gb, total_under_allocated_gb = 0.0, 0.0
jobs_under_allocated_count, total_jobs_processed = 0, 0


# --- Streaming loop ---
# We iterate through the pre-processed simulation_df
for i, row in simulation_df.iterrows():
    total_jobs_processed += 1

    # --- SIMPLIFIED: Prepare data for prediction ---
    # The 'simulation_df' already contains all the features in the correct format.
    # We just need to select them in the right order.
    input_features = pd.DataFrame([row[feature_columns_from_model]])

    # --- Run Prediction ---
    predicted_class = model.predict(input_features)[0]
    allocated_mem_gb = bin_edges_gb[min(
        predicted_class + 1, len(bin_edges_gb) - 1)]
    # The true value is in the 'max_rss_gb_true' column of our holdout file
    actual_gb = row["max_rss_gb_true"]
    difference_gb = allocated_mem_gb - actual_gb

    # Update cumulative trackers
    if difference_gb < 0:
        jobs_under_allocated_count += 1
        total_under_allocated_gb += abs(difference_gb)
    else:
        total_over_allocated_gb += difference_gb

    # --- Update UI elements ---
    with summary_placeholder.container():
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Jobs Processed", total_jobs_processed)
        under_alloc_percent = (jobs_under_allocated_count /
                               total_jobs_processed) * 100 if total_jobs_processed > 0 else 0
        sum_col2.metric("Jobs Under-allocated",
                        f"{jobs_under_allocated_count} ({under_alloc_percent:.1f}%)")
        sum_col3.metric("Total Under-allocated Shortfall",
                        f"{total_under_allocated_gb:.2f} GB")
        sum_col4.metric("Total Over-allocated Waste",
                        f"{total_over_allocated_gb:.2f} GB")

    results["Time"].append(row["time"])
    results["Allocated Memory (GB)"].append(allocated_mem_gb)
    results["Actual Memory (GB)"].append(actual_gb)

    with output_placeholder.container():
        st.markdown(f"**Timestamp:** `{row['time']}`")
        st.metric("Predicted Class", f"Class {predicted_class}")
        st.metric("Memory Allocated", f"{allocated_mem_gb:.2f} GB")
        st.metric("Actual Memory Used", f"{actual_gb:.2f} GB",
                  f"{difference_gb:.2f} GB", delta_color="inverse")
        st.caption(
            "Delta shows (Allocated - Actual). Negative is under-allocation.")

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
