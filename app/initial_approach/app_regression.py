import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque
from utils import (setup_sidebar, setup_ui, run_simulation_loop)

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
    payload = load_model_payload(MODEL_PAYLOAD_PATH)
    if not payload:
        st.stop()

    model = payload['model']
    model_name = payload['model_name']
    bin_edges_gb = payload['bin_edges_gb']
    features = payload['feature_columns']
    requires_one_hot = payload['requires_one_hot']

    try:
        df = pd.read_csv(SIMULATION_DATA_PATH, delimiter=";")
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values("time", inplace=True)
        if not requires_one_hot:
            for col in df.select_dtypes(include=['object']).columns:
                if col in features:
                    df[col] = df[col].astype('category')
    except FileNotFoundError:
        st.error(f"FATAL: Simulation data not found: '{SIMULATION_DATA_PATH}'")
        st.stop()

    delay_seconds = setup_sidebar(model_name, bin_edges_gb)
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, _):
        inp = pd.DataFrame([row[features]])
        pred_class = model.predict(inp)[0][0]
        alloc = bin_edges_gb[min(pred_class + 1, len(bin_edges_gb) - 1)]
        return alloc, pred_class

    run_simulation_loop(df, predict_fn,
                        actual_col="max_rss_gb_true",
                        memreq_col="memreq_gb",
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=True)

