import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import warnings
import ast
from collections import deque
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from resource_prediction.models import QEPredictor
import joblib
import time
from collections import deque
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import os
import pickle
from initial_approach.app_regression import run_regression
from classification.app_classification import run_classification
from utils import (setup_sidebar, setup_ui, run_simulation_loop)

def run_qe(MODEL_PAYLOAD_PATH):
    """Runs the app for the selected quantile ensemble model"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SIMULATION_DATA = os.path.join(SCRIPT_DIR, "qe/simulation_data.csv")

    # Load the model 
    try:
        # Add project root to Python path for proper imports
        import sys
        project_root = os.path.dirname(SCRIPT_DIR)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import model classes to make them available for unpickling
        from resource_prediction.models import QEPredictor, QuantileEnsemblePredictor
        
        model_path = os.path.join(SCRIPT_DIR, MODEL_PAYLOAD_PATH)
        model = joblib.load(model_path)
        
        # Handle backward compatibility: older models may have 'cols' instead of 'columns'
        if hasattr(model, 'cols') and not hasattr(model, 'columns'):
            model.columns = model.cols
            
    except Exception as e:
        st.error(f"FATAL: Could not load model. Error: {e}")
        st.stop()

    # Load the simulation data
    try:
        simulation_df = pd.read_csv(SIMULATION_DATA, delimiter=";")
        for col in ["bp_arch", "bp_compiler", "bp_opt", "component", "makeType",
                    "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_weekofyear"]:
            if col in simulation_df.columns:
                simulation_df[col] = simulation_df[col].astype("category")
    except FileNotFoundError:
        st.error("Please ensure training script has been run.")
        return None

    predictions = model.predict(simulation_df)

    # Streamlit setup
    delay_seconds = setup_sidebar("Quantile Ensemble")
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, idx):
        return predictions[idx], None  # no class for QE

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col="actual_max_rss_gb",
                        memreq_col="memreq_gb",
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=False)


# Define the models the user can choose from 
INITIAL_APPROACH = "Initial Approach - Classification"
CLASSIFICATION = "Classification"
QE_BALANCED = "Quantile-Ensemble - Balanced Approach"
QE_TINY_UNDER_ALLOC = "Quantile-Ensemble - Tiny Under Allocation"
QE_SMALL_WASTE = "Quantile-Ensemble - Small Memory Waste"

st.set_page_config(layout="wide")

# Initialize session state with default model
if "model_type" not in st.session_state:
    st.session_state.model_type = INITIAL_APPROACH

# Sidebar model selector
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose prediction model:", [INITIAL_APPROACH, CLASSIFICATION, QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE], 
                                index=0)

# If changed, update session state and rerun
if model_choice != st.session_state.model_type:
    st.session_state.model_type = model_choice
    st.rerun()

# Configuration - Updated paths to use absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATHS = {
    INITIAL_APPROACH: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/initial_approach/final_model.pkl"),
    CLASSIFICATION: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/classification/xgboost_uncertainty_model.pkl"),
    QE_BALANCED: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/qe/qe_balanced.pkl"),
    QE_TINY_UNDER_ALLOC: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/qe/qe_tiny_under_alloc.pkl"),
    QE_SMALL_WASTE: os.path.join(PROJECT_ROOT, "artifacts/trained_models/app/qe/qe_small_waste.pkl"),
}

MODEL_PAYLOAD_PATH = MODEL_PATHS.get(st.session_state.model_type, "../artifacts/trained_models/app/initial_approach/final_model.pkl")

# Depending on the model run the required function
if st.session_state.model_type == INITIAL_APPROACH:
    SIMULATION_DATA_PATH = "initial_approach/simulation_data.csv"
    run_regression(MODEL_PAYLOAD_PATH, SIMULATION_DATA_PATH)
elif st.session_state.model_type == CLASSIFICATION:
    run_classification(MODEL_PAYLOAD_PATH)
elif st.session_state.model_type in (QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE):
    run_qe(MODEL_PAYLOAD_PATH)
