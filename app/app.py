import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
import time
import os
import pickle
import sys
import ast
from collections import deque

# Add project root to Python path for proper imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from resource_prediction.models import QuantileEnsemblePredictor

# Now add the app directory to import app modules
APP_DIR = SCRIPT_DIR
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils import (setup_sidebar, setup_ui, run_simulation_loop)
from data_loader import load_unified_simulation_data, get_target_columns

def run_classification(model_path):
    """Runs the app for the classification model"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SUMMARY_FILE = os.path.join(SCRIPT_DIR, 'classification/best_strategy_summary_xgboost_10.csv')

    # Load the unified simulation data
    simulation_df = load_unified_simulation_data()
    if simulation_df is None:
        st.error("Failed to load simulation data. Please ensure data preprocessing has been completed.")
        st.stop()

    # Load the metadata
    try:
        summary_df = pd.read_csv(SUMMARY_FILE)
        best_info = summary_df.iloc[0]
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        BIN_EDGES_GB = ast.literal_eval(best_info['bin_edges'])
        params = ast.literal_eval(best_info['best_params_from_grid'])
        CONF_THRESH = params.get('confidence_threshold', 0.6)
        num_classes = 7
    except Exception as e:
        st.error(f"FATAL: Could not load model or config. Error: {e}")
        st.stop()

    # Make predictions using the model
    try:
        # Extract the features needed for classification model
        feature_cols = ["bp_arch", "bp_compiler", "bp_opt", "component", "makeType", 
                       "target_cnt", "lag_1_grouped", "rolling_p95_rss_g1_w5", 
                       "jobs", "localJobs"]
        
        # Filter to only the columns that exist in simulation_df and are needed
        available_cols = [col for col in feature_cols if col in simulation_df.columns]
        X_test = simulation_df[available_cols].copy()
        
        # Ensure categorical columns are properly typed
        for col in ["bp_arch", "bp_compiler", "bp_opt", "component", "makeType"]:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")

        y_pred_probs = model.predict_proba(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        preds = []
        for i, probs in enumerate(y_pred_probs):
            pred = y_pred_classes[i]
            if probs[pred] < CONF_THRESH:
                pred = min(pred + 1, num_classes - 1)
            preds.append(pred)
        simulation_df['predicted_class'] = preds
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # Get target column names
    target_cols = get_target_columns()

    # Streamlit setup
    delay_seconds = setup_sidebar("XGBoost Classification", BIN_EDGES_GB)
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, _):
        pred_class = row["predicted_class"]
        alloc = BIN_EDGES_GB[min(pred_class + 1, len(BIN_EDGES_GB) - 1)]
        return alloc, pred_class

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col=target_cols['actual_col'],
                        memreq_col=target_cols['memreq_col'],
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=True)


def run_qe(model_path):
    """Runs the app for the selected quantile ensemble model"""
    
    # Load the unified simulation data
    simulation_df = load_unified_simulation_data()
    if simulation_df is None:
        st.error("Failed to load simulation data. Please ensure data preprocessing has been completed.")
        st.stop()
    
    # Load the model 
    try:
        model_data = joblib.load(model_path)
        
        # Handle new Pareto models which are saved as dictionaries
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        # Handle backward compatibility: older models may have 'cols' instead of 'columns'
        if hasattr(model, 'cols') and not hasattr(model, 'columns'):
            model.columns = model.cols
            
    except Exception as e:
        st.error(f"FATAL: Could not load model. Error: {e}")
        st.stop()

    # Get predictions
    try:
        predictions = model.predict(simulation_df)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # Get target column names
    target_cols = get_target_columns()

    # Streamlit setup
    delay_seconds = setup_sidebar("Quantile Ensemble")
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, idx):
        return predictions[idx], None  # no class for QE

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col=target_cols['actual_col'],
                        memreq_col=target_cols['memreq_col'],
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=False)
    """Runs the app for the selected quantile ensemble model"""
    
    # Load the unified simulation data
    simulation_df = load_unified_simulation_data()
    if simulation_df is None:
        st.error("Failed to load simulation data. Please ensure data preprocessing has been completed.")
        st.stop()
    
    # Load the model 
    try:
        model = joblib.load(model_path)
        
        # Handle backward compatibility: older models may have 'cols' instead of 'columns'
        if hasattr(model, 'cols') and not hasattr(model, 'columns'):
            model.columns = model.cols
            
    except Exception as e:
        st.error(f"FATAL: Could not load model. Error: {e}")
        st.stop()

    # Get predictions
    try:
        predictions = model.predict(simulation_df)
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # Get target column names
    target_cols = get_target_columns()

    # Streamlit setup
    delay_seconds = setup_sidebar("Quantile Ensemble")
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, idx):
        return predictions[idx], None  # no class for QE

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col=target_cols['actual_col'],
                        memreq_col=target_cols['memreq_col'],
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=False)


# Define the models the user can choose from 
CLASSIFICATION = "Classification"
QE_BALANCED = "Quantile-Ensemble - Balanced Approach"
QE_TINY_UNDER_ALLOC = "Quantile-Ensemble - Tiny Under Allocation"
QE_SMALL_WASTE = "Quantile-Ensemble - Small Memory Waste"

st.set_page_config(layout="wide")

# Initialize session state with default model
if "model_type" not in st.session_state:
    st.session_state.model_type = CLASSIFICATION

# Sidebar model selector
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose prediction model:", [CLASSIFICATION, QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE], 
                                index=0)

# If changed, update session state and rerun
if model_choice != st.session_state.model_type:
    st.session_state.model_type = model_choice
    st.rerun()

# Configuration - Updated to use new Pareto frontier models
MODEL_PATHS = {
    CLASSIFICATION: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/classification/xgboost_uncertainty_model.pkl"),
    QE_BALANCED: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_balanced.pkl"),
    QE_TINY_UNDER_ALLOC: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_low_underallocation.pkl"),
    QE_SMALL_WASTE: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_low_waste.pkl"),
}

MODEL_PAYLOAD_PATH = MODEL_PATHS.get(st.session_state.model_type)

# Depending on the model run the required function
if st.session_state.model_type == CLASSIFICATION:
    run_classification(MODEL_PAYLOAD_PATH)
elif st.session_state.model_type in (QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE):
    run_qe(MODEL_PAYLOAD_PATH)
