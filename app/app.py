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
from resource_prediction.models import load_model


def run_unified_model(model_path, model_name, confidence_threshold=0.6):
    """Runs the app for any unified model type"""
    
    # Load the unified simulation data
    simulation_df = load_unified_simulation_data()
    if simulation_df is None:
        st.error("Failed to load simulation data. Please ensure data preprocessing has been completed.")
        st.stop()

    # Load the unified model
    try:
        unified_model = load_model(model_path)
        model_info = unified_model.get_model_info()
        
        # For classification models, extract bin edges for UI
        if unified_model.task_type == 'classification':
            BIN_EDGES_GB = model_info.get('bin_edges', [])
        else:
            # For regression models, create reasonable bin edges for UI display
            target_col = get_target_columns()['actual_col']
            if target_col in simulation_df.columns:
                min_val = simulation_df[target_col].min()
                max_val = simulation_df[target_col].max()
                BIN_EDGES_GB = np.linspace(min_val, max_val, 6).tolist()
            else:
                BIN_EDGES_GB = [0, 1, 2, 4, 8, 16]  # Default values
                
    except Exception as e:
        st.error(f"FATAL: Could not load model. Error: {e}")
        st.stop()

    # Make predictions using the unified interface
    try:
        # The unified model handles all preprocessing internally
        predictions = unified_model.predict(simulation_df, confidence_threshold=confidence_threshold)
        simulation_df['predictions'] = predictions
        
        # For classification models, also store predicted classes for display
        if unified_model.task_type == 'classification':
            # Reverse engineer class from allocation using bin edges
            bin_edges = np.array(unified_model.bin_edges)
            predicted_classes = []
            for pred in predictions:
                # Find which bin edge this prediction corresponds to
                class_idx = np.searchsorted(bin_edges[1:], pred, side='right')
                predicted_classes.append(class_idx)
            simulation_df['predicted_class'] = predicted_classes
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.stop()

    # Get target column names
    target_cols = get_target_columns()

    # Streamlit setup
    delay_seconds = setup_sidebar(model_name, BIN_EDGES_GB)
    summary_ph, output_ph, chart_ph = setup_ui(st.session_state.model_type)

    def predict_fn(row, _):
        """Prediction function for the simulation loop"""
        allocation = row["predictions"]
        predicted_class = row.get("predicted_class", 0) if unified_model.task_type == 'classification' else 0
        return allocation, predicted_class

    # Show classification classes only for classification models
    show_class = unified_model.task_type == 'classification'

    run_simulation_loop(simulation_df, predict_fn,
                        actual_col=target_cols['actual_col'],
                        memreq_col=target_cols['memreq_col'],
                        summary_placeholder=summary_ph,
                        output_placeholder=output_ph,
                        chart_placeholder=chart_ph,
                        delay_seconds=delay_seconds,
                        show_class=show_class)

# Legacy function kept for backward compatibility - now uses unified interface
def run_classification(model_path, model_name):
    """Runs the app for classification models (legacy interface)"""
    return run_unified_model(model_path, model_name, confidence_threshold=0.6)


# Legacy function kept for backward compatibility - now uses unified interface  
def run_qe(model_path):
    """Runs the app for QE models (legacy interface)"""
    model_name = f"QE Model ({model_path.split('/')[-1].replace('.pkl', '')})"
    return run_unified_model(model_path, model_name, confidence_threshold=0.6)


def main():
    """Main Streamlit application entry point"""
    
    st.set_page_config(
        page_title="Resource Prediction App",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Memory Allocation Prediction App")
    st.markdown("---")
    
    # Model selection in sidebar
    st.sidebar.title("Model Selection")
    
    # Available models configuration - using unified models
    available_models = {
        "LightGBM Classification": "artifacts/unified_models/unified_lightgbm_classification.pkl",
        "XGBoost Classification": "artifacts/unified_models/unified_xgboost_classification.pkl", 
        "QE Balanced": "artifacts/pareto/models/qe_balanced.pkl",
        "QE Low Waste": "artifacts/pareto/models/qe_low_waste.pkl",
        "QE Low Underallocation": "artifacts/pareto/models/qe_low_underallocation.pkl"
    }
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(available_models.keys()),
        index=0
    )
    
    # Store model selection in session state
    if "model_type" not in st.session_state:
        st.session_state.model_type = selected_model
    
    if st.session_state.model_type != selected_model:
        st.session_state.model_type = selected_model
        st.rerun()
    
    # Get model path
    model_path = os.path.join(PROJECT_ROOT, available_models[selected_model])
    
    # Display model information
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Selected Model:** {selected_model}")
    
    try:
        # Load model info for display
        unified_model = load_model(model_path)
        model_info = unified_model.get_model_info()
        
        st.sidebar.markdown(f"**Model Type:** {model_info['model_type']}")
        st.sidebar.markdown(f"**Task Type:** {model_info['task_type']}")
        st.sidebar.markdown(f"**Features:** {model_info['num_features']}")
        
        if 'num_classes' in model_info:
            st.sidebar.markdown(f"**Classes:** {model_info['num_classes']}")
            
    except Exception as e:
        st.sidebar.error(f"Error loading model info: {e}")
    
    # Run the selected model
    try:
        run_unified_model(model_path, selected_model)
    except Exception as e:
        st.error(f"Error running model: {e}")
        st.stop()


if __name__ == "__main__":
    main()
