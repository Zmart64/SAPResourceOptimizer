import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to Python path for proper imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now add the app directory to import app modules
APP_DIR = SCRIPT_DIR
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils import (setup_sidebar, setup_ui, run_simulation_loop)
from data_loader import load_unified_simulation_data, get_target_columns

def run_classification(model_path, model_name):
    """Runs the app for the classification model"""
    
    # Load the unified simulation data
    simulation_df = load_unified_simulation_data()
    if simulation_df is None:
        st.error("Failed to load simulation data. Please ensure data preprocessing has been completed.")
        st.stop()

    # Load the new model (dictionary format)
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        BIN_EDGES_GB = model_data['bin_edges'].tolist()
        model_features = model_data['features']
        num_classes = len(model.classes_)
        CONF_THRESH = 0.6  # Default confidence threshold
    except Exception as e:
        st.error(f"FATAL: Could not load model or config. Error: {e}")
        st.stop()

    # Make predictions using the model
    try:
        # Prepare features for prediction by handling one-hot encoding
        X_test = simulation_df.copy()
        
        # Apply one-hot encoding for categorical features that the model expects
        categorical_features = ['location', 'component', 'makeType', 'bp_arch', 'bp_compiler', 'bp_opt']
        
        for cat_feature in categorical_features:
            if cat_feature in X_test.columns:
                # Get dummies for the categorical feature
                dummies = pd.get_dummies(X_test[cat_feature], prefix=cat_feature, dtype=int)
                
                # Add dummy columns to X_test
                for dummy_col in dummies.columns:
                    X_test[dummy_col] = dummies[dummy_col]
        
        # Add feature mapping for compatibility
        if 'lag_max_rss_g1_w1' in X_test.columns and 'lag_max_rss_global_w5' not in X_test.columns:
            X_test['lag_max_rss_global_w5'] = X_test['lag_max_rss_g1_w1']
        
        # Handle missing one-hot encoded features by creating them with zeros
        for feature in model_features:
            if feature not in X_test.columns:
                # Check if it's a one-hot encoded categorical feature
                for cat_prefix in ['location_', 'component_', 'makeType_', 'bp_arch_', 'bp_compiler_', 'bp_opt_']:
                    if feature.startswith(cat_prefix):
                        X_test[feature] = 0
                        break
        
        # Select features that are available in both model and processed data
        available_features = [f for f in model_features if f in X_test.columns]
        
        print(f"Using {len(available_features)}/{len(model_features)} model features")
        
        if len(available_features) < len(model_features) * 0.5:  # Require at least 50% of features
            st.error(f"Too few matching features: {len(available_features)}/{len(model_features)}")
            st.stop()
        
        # Select only the features the model expects and ensure correct order
        X_test_model = X_test[model_features].copy()
        
        # Convert categorical columns to numeric if needed for prediction
        for col in X_test_model.columns:
            if X_test_model[col].dtype.name == 'category':
                X_test_model[col] = X_test_model[col].astype(float)
        
        X_test_model = X_test_model.fillna(0)
        
        # Make predictions
        y_pred_probs = model.predict_proba(X_test_model)
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
    delay_seconds = setup_sidebar(model_name, BIN_EDGES_GB)
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


# Define the models the user can choose from 
CLASSIFICATION_LIGHTGBM = "LightGBM Classification"
CLASSIFICATION_XGBOOST = "XGBoost Classification"
QE_BALANCED = "Quantile-Ensemble - Balanced Approach"
QE_TINY_UNDER_ALLOC = "Quantile-Ensemble - Tiny Under Allocation"
QE_SMALL_WASTE = "Quantile-Ensemble - Small Memory Waste"

st.set_page_config(layout="wide")

# Initialize session state with default model
if "model_type" not in st.session_state:
    st.session_state.model_type = CLASSIFICATION_LIGHTGBM

# Sidebar model selector
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose prediction model:", 
                                [CLASSIFICATION_LIGHTGBM, CLASSIFICATION_XGBOOST, QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE], 
                                index=0)

# If changed, update session state and rerun
if model_choice != st.session_state.model_type:
    st.session_state.model_type = model_choice
    st.rerun()

# Configuration - Updated to use new classification models and Pareto frontier QE models
MODEL_PATHS = {
    CLASSIFICATION_LIGHTGBM: os.path.join(PROJECT_ROOT, "artifacts/trained_models/lightgbm_classification.pkl"),
    CLASSIFICATION_XGBOOST: os.path.join(PROJECT_ROOT, "artifacts/trained_models/xgboost_classification.pkl"),
    QE_BALANCED: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_balanced.pkl"),
    QE_TINY_UNDER_ALLOC: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_low_underallocation.pkl"),
    QE_SMALL_WASTE: os.path.join(PROJECT_ROOT, "artifacts/pareto/models/qe_low_waste.pkl"),
}

MODEL_PAYLOAD_PATH = MODEL_PATHS.get(st.session_state.model_type)

# Depending on the model run the required function
if st.session_state.model_type == CLASSIFICATION_LIGHTGBM:
    run_classification(MODEL_PAYLOAD_PATH, "LightGBM Classification")
elif st.session_state.model_type == CLASSIFICATION_XGBOOST:
    run_classification(MODEL_PAYLOAD_PATH, "XGBoost Classification")
elif st.session_state.model_type in (QE_BALANCED, QE_TINY_UNDER_ALLOC, QE_SMALL_WASTE):
    run_qe(MODEL_PAYLOAD_PATH)
