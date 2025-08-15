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
import sys
from collections import deque

# Add the app directory to the Python path to import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(SCRIPT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils import (setup_sidebar, setup_ui, run_simulation_loop)


# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def run_classification(MODEL_PAYLOAD_PATH):
    """Runs the app for the classification model"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SUMMARY_FILE = os.path.join(SCRIPT_DIR, 'best_strategy_summary_xgboost_10.csv')
    FULL_FILE = os.path.join(SCRIPT_DIR, "test_F_and_y.csv")
    TEST_FILE = os.path.join(SCRIPT_DIR, "test_X.csv")

    # Load the metadata
    try:
        summary_df = pd.read_csv(SUMMARY_FILE)
        best_info = summary_df.iloc[0]
        model_path = os.path.join(MODEL_PAYLOAD_PATH)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        BIN_EDGES_GB = ast.literal_eval(best_info['bin_edges'])
        params = ast.literal_eval(best_info['best_params_from_grid'])
        CONF_THRESH = params.get('confidence_threshold', 0.6)
        num_classes = 7
    except Exception as e:
        st.error(f"FATAL: Could not load model or config. Error: {e}")
        st.stop()

    try:
        F_with_y = pd.read_csv(FULL_FILE, delimiter=";")
        X_test = pd.read_csv(TEST_FILE, delimiter=";")
        for col in ["bp_arch", "bp_compiler", "bp_opt", "component", "makeType"]:
            X_test[col] = X_test[col].astype("category")
    except FileNotFoundError:
        st.error("Please ensure training script has been run.")
        return None

    simulation_df = F_with_y.copy()

    y_pred_probs = model.predict_proba(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    preds = []
    for i, probs in enumerate(y_pred_probs):
        pred = y_pred_classes[i]
        if probs[pred] < CONF_THRESH:
            pred = min(pred + 1, num_classes - 1)
        preds.append(pred)
    simulation_df['predicted_class'] = preds

    # Streamlit setup
    delay_seconds = setup_sidebar("XGBoost Classification", BIN_EDGES_GB)
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, _):
        pred_class = row["predicted_class"]
        alloc = BIN_EDGES_GB[min(pred_class + 1, len(BIN_EDGES_GB) - 1)]
        return alloc, pred_class

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col="max_rss_gb",
                        memreq_col="memreq",
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=True)

