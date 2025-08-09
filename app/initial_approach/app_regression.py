import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque

@st.cache_resource
def load_model_payload(model_path):
    """Loads the entire model payload (model, features, bins, etc.) from the pkl file."""
    try:
        payload = joblib.load(model_path)
        return payload
    except FileNotFoundError:
        st.error(
            f"FATAL: Model payload file not found at '{model_path}'.")
        st.error("Please run the training script first to generate the model file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model payload: {e}")
        return None

def run_regression(MODEL_PAYLOAD_PATH, SIMULATION_DATA_PATH):

    model_payload = load_model_payload(MODEL_PAYLOAD_PATH)
    if not model_payload:
        st.stop()

    # Extract all components from the rich payload
    model = model_payload['model']
    model_name = model_payload['model_name']
    bin_edges_gb = model_payload['bin_edges_gb']
    feature_columns_from_model = model_payload['feature_columns']
    requires_one_hot = model_payload['requires_one_hot']
    original_categorical_features = model_payload['original_categorical_features']

    @st.cache_data
    def load_simulation_data(file_path):
        """Loads the pre-processed simulation data."""
        try:
            df = pd.read_csv(file_path, delimiter=";")
        except FileNotFoundError:
            st.error(f"FATAL: Simulation data file not found at '{file_path}'.")
            st.error(
                "Please ensure the training script has been run and the holdout file was created.")
            return None

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values("time").reset_index(drop=True)

        if not requires_one_hot:
            for col in df.select_dtypes(include=['object']).columns:
                if col in feature_columns_from_model:
                    df[col] = df[col].astype('category')

        return df

    simulation_df = load_simulation_data(SIMULATION_DATA_PATH)
    if simulation_df is None:
        st.stop()


    # --- 3. Streamlit App ---
    st.title(f"Real-time Allocation Simulator ({st.session_state.model_type})")

    # --- Sidebar ---
    st.sidebar.header("Simulation Controls")
    delay_seconds = st.sidebar.slider(
        "Delay between steps (seconds)", 0.0, 2.0, 0.1)
    st.sidebar.subheader("Loaded Model Info")
    st.sidebar.markdown(f"**Model Type:** `{model_name}`")
    st.sidebar.write("**Memory Bins (GB):**")
    st.sidebar.code(np.round(bin_edges_gb, 2))

    # --- Placeholders for UI elements ---
    st.subheader("Overall Simulation Summary")
    summary_placeholder = st.empty()
    st.write("### Streaming Predictions (Live Sliding Window)")
    col1, col2 = st.columns([1, 2])
    output_placeholder = col1.empty()
    chart_placeholder = col2.empty()

    # --- Initialize deques and cumulative trackers ---
    window_size = 30
    results = {"Time": deque(maxlen=window_size), "Allocated Memory (GB)": deque(
        maxlen=window_size), "Actual Memory (GB)": deque(maxlen=window_size)}
    total_over_allocated_gb, total_under_allocated_gb , total_saved_memory, total_allocated_gb, total_actual_memory = 0.0, 0.0, 0.0, 0.0, 0.0
    jobs_under_allocated_count, total_jobs_processed = 0, 0


    # --- Streaming loop ---
    for i, row in simulation_df.iterrows():
        total_jobs_processed += 1

        # Prepare data for prediction
        input_features = pd.DataFrame([row[feature_columns_from_model]])

        # --- Run Prediction ---
        # *** THIS IS THE CORRECTED LINE ***
        predicted_class = model.predict(input_features)[0][0]

        allocated_mem_gb = bin_edges_gb[min(
            predicted_class + 1, len(bin_edges_gb) - 1)]
        actual_gb = row["max_rss_gb_true"]
        difference_gb = allocated_mem_gb - actual_gb

        # Update cumulative trackers
        if difference_gb < 0:
            jobs_under_allocated_count += 1
            total_under_allocated_gb += abs(difference_gb)
        else:
            total_over_allocated_gb += difference_gb
            total_actual_memory += actual_gb
            total_allocated_gb += allocated_mem_gb
            # Only update the saved memory if an underallocation DID NOT happen
            total_saved_memory += row["memreq_gb"] - allocated_mem_gb

        # --- Update UI elements ---
        with summary_placeholder.container():
            sum_col1, sum_col2, sum_col3, sum_col4, sum_col5, sum_col6 = st.columns(6)
            sum_col1.metric("Total Jobs Processed", total_jobs_processed)
            under_alloc_percent = (jobs_under_allocated_count /
                                total_jobs_processed) * 100 if total_jobs_processed > 0 else 0
            over_alloc_percent = (total_allocated_gb /
                                total_actual_memory) * 100 - 100 if total_actual_memory > 0.0 else 0
            sum_col2.metric("Jobs Under-allocated",
                            f"{jobs_under_allocated_count} ({under_alloc_percent:.1f}%)")
            sum_col3.metric("Total Under-allocated Shortfall",
                            f"{total_under_allocated_gb:.2f} GB")
            sum_col4.metric("Total Over-allocated Waste",
                            f"{total_over_allocated_gb:.2f} GB")
            sum_col5.metric("Overallocation",
                            f"{over_alloc_percent:.1f}%")
            sum_col6.metric(f"Total Saved Memory",
                            f"{total_saved_memory:.2f} GB")

        results["Time"].append(row["time"])
        results["Allocated Memory (GB)"].append(allocated_mem_gb)
        results["Actual Memory (GB)"].append(actual_gb)

        with output_placeholder.container():
            st.markdown(f"**Timestamp:** `{row['time']}`")
            st.metric("Predicted Class", f"Class {predicted_class}")
            st.metric("Memory Allocated", f"{allocated_mem_gb:.2f} GB")
            st.metric("Actual Memory Used", f"{actual_gb:.2f} GB",
                    f"{difference_gb:.2f} GB")
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
