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
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import os
import pickle

class QEPredictor:
    def __init__(self, *, alpha, safety, gb_p, xgb_p, seed=42):
        self.alpha, self.safety = alpha, safety
        self.gb  = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, random_state=seed, **gb_p)
        base = dict(objective="reg:quantileerror",
                    quantile_alpha=alpha, n_jobs=1, random_state=seed)
        base.update(xgb_p)
        self.xgb = xgb.XGBRegressor(**base)
        self.cols = None

    def _enc(self, X, fit=False):
        Xd = pd.get_dummies(X, drop_first=True)
        if Xd.columns.duplicated().any():
            Xd = Xd.loc[:, ~Xd.columns.duplicated()]
        if fit:
            self.cols = Xd.columns
        else:
            miss = self.cols.difference(Xd.columns)
            for c in miss: Xd[c] = 0
            Xd = Xd[self.cols]
        return Xd.astype(float)

    def fit(self, X, y):
        self.gb.fit(self._enc(X, True), y)
        self.xgb.fit(self._enc(X), y, verbose=False)

    def predict(self, X):
        Xd = self._enc(X, False)
        p  = np.maximum(self.gb.predict(Xd), self.xgb.predict(Xd))
        return p * self.safety


# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def run_qe(MODEL_PAYLOAD_PATH):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SIMULATION_DATA = os.path.join(SCRIPT_DIR, "simulation_data.csv")
    
    try:
        model_path = os.path.join(SCRIPT_DIR, MODEL_PAYLOAD_PATH)
        model = joblib.load(model_path) 

    except Exception as e:
        print(f"FATAL: Could not load model or its configuration. Error: {e}")
        exit()
    
    # --- Load the pre-processed simulation data 
    try:
        simulation_df = pd.read_csv(SIMULATION_DATA, delimiter=";")

        categorical_cols = [
        "bp_arch", "bp_compiler", "bp_opt",
        "component", "makeType",
        "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_weekofyear"
    ]

        for col in categorical_cols:
            if col in simulation_df.columns:
                simulation_df[col] = simulation_df[col].astype("category")

    except FileNotFoundError:
        st.error("Please ensure the training script has been run and the holdout file was created.")
        return None

    if simulation_df is None:
        st.stop()

    # --- Prediction ---
    predictions = model.predict(simulation_df)

    # --- Streamlit App ---
    st.title(f"Real-time Allocation Simulator ({st.session_state.model_type})")

    # --- Sidebar ---
    st.sidebar.header("Simulation Controls")
    delay_seconds = st.sidebar.slider(
        "Delay between steps (seconds)", 0.0, 2.0, 0.1)
    st.sidebar.info("This app simulates real-time data arriving one row at a time. The loaded model predicts a memory 'class', which is then translated into a GB allocation.")
    st.sidebar.subheader("Loaded Model Info")
    st.sidebar.markdown(f"**Model Type:** `Quantile Ensemble`")

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
    total_over_allocated_gb, total_under_allocated_gb , total_saved_memory = 0.0, 0.0, 0.0
    jobs_under_allocated_count, total_jobs_processed = 0, 0


    # --- Streaming loop ---
    for i, row in simulation_df.iterrows():
        total_jobs_processed += 1

        allocated_mem_gb = predictions[i]
        actual_gb = row["lag_max_rss_g1_w1"]
        difference_gb = allocated_mem_gb - actual_gb

        # Update cumulative trackers
        if difference_gb < 0:
            jobs_under_allocated_count += 1
            total_under_allocated_gb += abs(difference_gb)
        else:
            total_over_allocated_gb += difference_gb

            # Only update the saved memory if an underallocation DID NOT happen
            total_saved_memory += row["memreq_gb"] - allocated_mem_gb

        # --- Update UI elements ---
        with summary_placeholder.container():
            sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
            sum_col1.metric("Total Jobs Processed", total_jobs_processed)
            under_alloc_percent = (jobs_under_allocated_count /
                                total_jobs_processed) * 100 if total_jobs_processed > 0 else 0
            sum_col2.metric("Jobs Under-allocated",
                            f"{jobs_under_allocated_count} ({under_alloc_percent:.1f}%)")
            sum_col3.metric("Total Under-allocated Shortfall",
                            f"{total_under_allocated_gb:.2f} GB")
            sum_col4.metric("Total Over-allocated Waste",
                            f"{total_over_allocated_gb:.2f} GB")
            sum_col5.metric(f"Total Saved Memory",
                            f"{total_saved_memory:.2f} GB")

        results["Time"].append(row["time"])
        results["Allocated Memory (GB)"].append(allocated_mem_gb)
        results["Actual Memory (GB)"].append(actual_gb)

        with output_placeholder.container():
            st.markdown(f"**Timestamp:** `{row['time']}`")
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

# --- 1. Load Model and All Necessary Metadata ---
st.set_page_config(layout="wide")

# Initialize session state with default model
if "model_type" not in st.session_state:
    st.session_state.model_type = "Regression"

# Sidebar model selector
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose prediction model:", ["Regression", "Classification", "Quantile-Ensemble"], 
                                index=0)

# If changed, update session state and rerun
if model_choice != st.session_state.model_type:
    st.session_state.model_type = model_choice
    st.rerun()

# --- Configuration ---
MODEL_PATHS = {
    "Regression": "regression/final_model.pkl",
    "Classification": "classification/xgboost_uncertainty_model.pkl",
    "Quantile-Ensemble": "qe_trial_32_55.pkl"
}

MODEL_PAYLOAD_PATH = MODEL_PATHS.get(st.session_state.model_type, "regression/final_model.pkl")

if st.session_state.model_type == "Quantile-Ensemble":
    run_qe(MODEL_PAYLOAD_PATH)

