import os
import time
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from collections import deque

def setup_sidebar(model_type, bin_edges_gb=None):
    """
    Sets up the Streamlit sidebar controls and displays model metadata.
    """
    st.sidebar.header("Simulation Controls")
    delay_seconds = st.sidebar.slider("Delay between steps (seconds)", 0.0, 2.0, 0.1)
    st.sidebar.subheader("Loaded Model Info")
    st.sidebar.markdown(f"**Model Type:** `{model_type}`")
    if bin_edges_gb is not None:
        st.sidebar.write("**Memory Bins (GB):**")
        st.sidebar.code(np.round(bin_edges_gb, 2))
    return delay_seconds

def setup_ui(model_type):
    """
    Sets up the main Streamlit UI for the simulation view.
    """
    st.title(f"Real-time Allocation Simulator ({model_type})")
    st.subheader("Overall Simulation Summary")
    summary_placeholder = st.empty()
    st.write("### Streaming Predictions (Live Sliding Window)")
    col1, col2 = st.columns([1, 2])
    return summary_placeholder, col1.empty(), col2.empty()

def update_summary(placeholder, total_jobs, jobs_under, total_under, total_over, total_alloc, total_actual, total_saved):
    """
    Updates the summary metrics section with the latest cumulative simulation results.
    """
    with placeholder.container():
        cols = st.columns(6)
        under_pct = (jobs_under / total_jobs) * 100 if total_jobs else 0
        over_pct = (total_alloc / total_actual) * 100 - 100 if total_actual > 0 else 0
        cols[0].metric("Total Jobs Processed", total_jobs)
        cols[1].metric("Jobs Under-allocated", f"{jobs_under} ({under_pct:.1f}%)")
        cols[2].metric("Total Under-allocated Shortfall", f"{total_under:.2f} GB")
        cols[3].metric("Total Over-allocated Waste", f"{total_over:.2f} GB")
        cols[4].metric("Overallocation", f"{over_pct:.1f}%")
        cols[5].metric(f"Total Saved Memory", f"{total_saved:.2f} GB")

def update_output(placeholder, timestamp, predicted_class, allocated_mem_gb, actual_gb, diff_gb, show_class=True):
    """
    Updates the live job details section with current prediction and memory usage info.
    """
    with placeholder.container():
        st.markdown(f"**Timestamp:** `{timestamp}`")
        if show_class and predicted_class is not None:
            st.metric("Predicted Class", f"Class {predicted_class}")
        st.metric("Memory Allocated", f"{allocated_mem_gb:.2f} GB")
        st.metric("Actual Memory Used", f"{actual_gb:.2f} GB", f"{diff_gb:.2f} GB")
        st.caption("Delta shows (Allocated - Actual). Negative is under-allocation.")

def update_chart(placeholder, results):
    """
    Renders or updates the line chart comparing allocated vs. actual memory usage.
    """
    with placeholder.container():
        plot_df = pd.DataFrame(results)
        melted_df = plot_df.melt(id_vars="Time", var_name="Series", value_name="Memory (GB)")
        line_chart = alt.Chart(melted_df).mark_line(point=alt.OverlayMarkDef(size=50)).encode(
            x=alt.X("Time:T", title="Time"),
            y=alt.Y("Memory (GB):Q", title="Memory (GB)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title="Data Series",
                            scale=alt.Scale(domain=["Allocated Memory (GB)", "Actual Memory (GB)"],
                                            range=["#1f77b4", "#d62728"])),
            tooltip=["Time:T", "Series:N", alt.Tooltip("Memory (GB):Q", format=".2f")]
        ).properties(title="Allocated vs. Actual Memory Usage", height=400).interactive()
        st.altair_chart(line_chart, use_container_width=True)

def run_simulation_loop(simulation_df, predict_fn, actual_col, memreq_col,
                        summary_placeholder, output_placeholder, chart_placeholder, delay_seconds,
                        show_class=True):
    """
    Runs the main simulation loop, processing each job row-by-row and updating the UI.
    """
    window_size = 30
    results = {"Time": deque(maxlen=window_size),
               "Allocated Memory (GB)": deque(maxlen=window_size),
               "Actual Memory (GB)": deque(maxlen=window_size)}
    total_over, total_under, total_saved, total_alloc, total_actual = 0.0, 0.0, 0.0, 0.0, 0.0
    jobs_under, total_jobs = 0, 0

    for i, row in simulation_df.iterrows():
        total_jobs += 1
        allocated_mem_gb, predicted_class = predict_fn(row, i)
        actual_gb = row[actual_col]
        diff_gb = allocated_mem_gb - actual_gb

        if diff_gb < 0:
            jobs_under += 1
            total_under += abs(diff_gb)
        else:
            total_over += diff_gb
            total_actual += actual_gb
            total_alloc += allocated_mem_gb
            total_saved += row[memreq_col] - allocated_mem_gb

        update_summary(summary_placeholder, total_jobs, jobs_under, total_under, total_over, total_alloc, total_actual, total_saved)
        results["Time"].append(row["time"])
        results["Allocated Memory (GB)"].append(allocated_mem_gb)
        results["Actual Memory (GB)"].append(actual_gb)
        update_output(output_placeholder, row["time"], predicted_class, allocated_mem_gb, actual_gb, diff_gb, show_class)
        update_chart(chart_placeholder, results)

        time.sleep(delay_seconds)
