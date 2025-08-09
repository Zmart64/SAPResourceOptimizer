import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
import os
import warnings
import pickle
import ast
from collections import deque


# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def run_classification(MODEL_PAYLOAD_PATH):
    # --- Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Paths to the artifacts created by the training script
    SUMMARY_FILE = os.path.join(SCRIPT_DIR, 'best_strategy_summary_xgboost_10.csv')
    SUMMARY_FILE = os.path.normpath(SUMMARY_FILE)
    FULL_FILE = os.path.join(SCRIPT_DIR, "test_F_and_y.csv")
    TEST_FILE = os.path.join(SCRIPT_DIR, "test_X.csv")
    try:
        summary_df = pd.read_csv(SUMMARY_FILE)
        best_performer_info = summary_df.iloc[0]
        model_path = os.path.join(MODEL_PAYLOAD_PATH)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        BIN_EDGES_GB = ast.literal_eval(best_performer_info['bin_edges'])
        params = ast.literal_eval(best_performer_info['best_params_from_grid'])
        CONFIDENCE_THRESHOLD = params.get('confidence_threshold', 0.6)
        num_target_classes = 7

    except Exception as e:
        print(f"FATAL: Could not load model or its configuration. Error: {e}")
        exit()

    
    # --- Load the pre-processed simulation data and the general data
    try:
        F_with_y = pd.read_csv(FULL_FILE, delimiter=";")
        X_test = pd.read_csv(TEST_FILE, delimiter=";")

        categorical_cols = ["bp_arch", "bp_compiler", "bp_opt", "component", "makeType"]

        for col in categorical_cols:
            X_test[col] = X_test[col].astype("category")

    except FileNotFoundError:
        st.error("Please ensure the training script has been run and the holdout file was created.")
        return None
    simulation_df = F_with_y.copy()
    simulation_df['predicted_class'] = None

    if simulation_df is None:
        st.stop()

    # --- Generate Uncertainty-Aware Predictions ---
    y_pred_probs = model.predict_proba(X_test)
    y_pred_from_probs = np.argmax(y_pred_probs, axis=1)

    y_pred_probabilistic = []
    for i in range(len(y_pred_probs)):
        pred_class = y_pred_from_probs[i]
        confidence = y_pred_probs[i][pred_class]
        if confidence < CONFIDENCE_THRESHOLD:
            final_class = min(pred_class + 1, num_target_classes - 1)
        else:
            final_class = pred_class
        y_pred_probabilistic.append(final_class)

    simulation_df['predicted_class'] = y_pred_probabilistic

    # --- Streamlit App ---
    st.title(f"Real-time Allocation Simulator ({st.session_state.model_type})")

    # --- Sidebar ---
    st.sidebar.header("Simulation Controls")
    delay_seconds = st.sidebar.slider(
        "Delay between steps (seconds)", 0.0, 2.0, 0.1)
    st.sidebar.subheader("Loaded Model Info")
    st.sidebar.markdown(f"**Model Type:** `XGBoost Classification`")
    st.sidebar.write("**Memory Bins (GB):**")
    st.sidebar.code(np.round(BIN_EDGES_GB, 2))

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

        # Instead of predicting live, use precomputed prediction
        predicted_class = row["predicted_class"]
        # Use the upper edge of the predicted bin (or the last edge if out of range)
        allocated_mem_gb = BIN_EDGES_GB[min(predicted_class + 1, len(BIN_EDGES_GB) - 1)]
        actual_gb = row["max_rss_gb"]
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
            total_saved_memory += row["memreq"] / 1024 - allocated_mem_gb 

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
