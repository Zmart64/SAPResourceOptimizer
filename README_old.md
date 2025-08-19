# Resource Prediction

This repository contains a machine learning pipeline for predicting peak memory 
usage of distributed build jobs. By analyzing historical build telemetry, the 
system can optimize memory allocation to reduce both out-of-memory failures 
and resource waste.

## Key Features

- **Multiple ML Approaches**: Regression and classification models for memory prediction
- **Automated Hyperparameter Tuning**: Optuna-based optimization with parallel execution  
- **Business-Focused Metrics**: Cost-aware objective function balancing failures vs waste
- **Rich Feature Engineering**: Temporal, categorical, and rolling window features
- **Production-Ready Pipeline**: Complete preprocessing, training, and evaluation workflow
- **Interactive Web Applications**: Streamlit-based dashboards for model exploration

## Quick Start

### Prerequisites

Install [Poetry](https://python-poetry.org/) for dependency management:

```console
# Install Poetry (macOS/Linux)
curl -sSL https://install.python-poetry.org | python3 -

# Or via pip  
pip install poetry
```

### Setup

```console
# Install dependencies
poetry install

# Activate the Poetry shell
poetry shell

# Or run commands with poetry run prefix
poetry run python --version
```

### Using Models

The project provides a `DeployableModel` wrapper for consistent model loading and prediction:

```python
from resource_prediction.models import DeployableModel

# Load any model type
model = DeployableModel.load("artifacts/trained_models/lightgbm_classification.pkl")

# Make predictions (preprocessing happens automatically)  
predictions = model.predict(raw_dataframe)

# Get model information
model_info = model.get_model_info()
print(f"Model: {model_info['model_type']} ({model_info['task_type']})")
```

### Documentation

Generate and serve documentation locally:

```console
# Build the documentation
make html

# Serve documentation (opens in browser)
make serve  

# Clean, build, and serve (for development)
make dev
```

## Imagined "how to use" workflow

### 1. Simple Training (Recommended)

```console
# Activate Poetry environment
poetry shell

# Train all models with default parameters (simple and fast)
python main.py --train

# Train only classification models
python main.py --train --task-type classification

# Train specific model families
python main.py --train --model-families xgboost_classification lightgbm_regression
```

### 2. Advanced Training with Hyperparameter Search

```console
# Run full pipeline with hyperparameter optimization for classification
python main.py --model-type classification --run-search

# Now regression, but re-use preprocessed data from first run  
python main.py --model-type regression --skip-preprocessing --run-search

# Quick training with defaults (equivalent to --train)
python main.py --run-search --use-defaults
```

### 3. Evaluation Only

```console
# Skip training and just evaluate existing models
python main.py --evaluate-only
```

## Command Line Reference

### Main Training Options

| Command | Description | Use Case |
|---------|-------------|----------|
| `--train` | Train models with default parameters | Quick prototyping, baseline results |
| `--run-search` | Full hyperparameter optimization | Best performance, production models |
| `--run-search --use-defaults` | Train with defaults (same as `--train`) | Legacy compatibility |
| `--evaluate-only` | Evaluate existing trained models | Testing, comparison |

### Common Filtering Options

| Option | Description | Example |
|--------|-------------|---------|
| `--task-type` | Filter by regression or classification | `--task-type regression` |
| `--model-families` | Train specific model families | `--model-families xgboost_regression lightgbm_classification` |
| `--skip-preprocessing` | Reuse preprocessed data | `--skip-preprocessing --train` |

### Quick Examples

```console
# Simple: Train all models with defaults
python main.py --train

# Focused: Train only XGBoost models  
python main.py --train --model-families xgboost_regression xgboost_classification

# Fast: Reuse preprocessing from previous run
python main.py --train --skip-preprocessing

# Advanced: Full hyperparameter search for specific models
python main.py --run-search --model-families lightgbm_regression
```

## Features

- **Multiple ML Approaches**: Regression and classification models for memory prediction
- **Flexible Training Options**: Simple `--train` mode or advanced `--run-search` with hyperparameter optimization
- **Automated Hyperparameter Tuning**: Optuna-based optimization with parallel execution
- **Business-Focused Metrics**: Cost-aware objective function balancing failures vs waste
- **Rich Feature Engineering**: Temporal, categorical, and rolling window features
- **Unified Model Architecture**: Standardized BasePredictor interface for all models
- **Interactive Web Applications**: Streamlit-based dashboards for model exploration
- **Production-Ready**: Poetry dependency management, comprehensive testing, and documentation

### Training Modes

**Simple Training (`--train`)**:
- Train models with default parameters (no hyperparameter search)
- 90% simpler than `--run-search --use-defaults`
- Perfect for quick prototyping and baseline results
- Example: `python main.py --train --model-families xgboost_regression`

**Advanced Training (`--run-search`)**:
- Full hyperparameter optimization using Optuna
- Best model performance but slower execution
- Use `--use-defaults` to skip search and use default parameters
- Example: `python main.py --run-search --model-families xgboost_regression`

## Supported Models

**Regression Models** (predict exact memory values):
- Quantile Ensemble (GradientBoosting + XGBoost)
- XGBoost Regression
- LightGBM Regression

**Classification Models** (predict memory bins):
- XGBoost Classifier
- LightGBM Classifier  
- Random Forest Classifier
- Logistic Regression

## Architecture

The pipeline implements a business-focused optimization objective:

**Business Score = 5 × Under-allocation% + Over-allocation%**

This reflects that memory under-allocation (causing build failures) is 5x more 
costly than over-allocation (wasting resources).

### Model Architecture

The project uses a clean, consistent model architecture:

- **BasePredictor Interface**: All models implement consistent `fit()` and `predict()` methods
- **DeployableModel Wrapper**: Production-ready wrapper with integrated preprocessing
- **ModelPreprocessor**: Handles feature engineering automatically
- **Unified Parameter Handling**: Both hyperparameter search and evaluation use the same wrapper models

### Directory Structure

The models are organized for clarity and maintainability:

```
resource_prediction/models/
├── __init__.py              # Public API exports
├── base.py                  # BasePredictor interface
├── unified_wrapper.py       # DeployableModel for production
└── implementations/         # Specific model implementations
    ├── lightgbm_models.py
    ├── quantile_ensemble.py
    ├── sklearn_models.py
    └── xgboost_models.py
```

**Design Principles:**
- **Separation of Concerns**: Infrastructure files (`base.py`, `unified_wrapper.py`) are separated from specific implementations
- **Clear Organization**: All concrete model implementations are grouped in the `implementations/` directory
- **Simple Imports**: Users import from `resource_prediction.models` regardless of internal structure
- **Easy Extension**: New models go in `implementations/` with import added to main `__init__.py`

## How to Extend

### Adding New Models

The system uses a simplified model registration approach that requires only 3 steps to add new models. The dynamic import and instantiation system automatically handles model creation without requiring hardcoded logic.

**Step-by-Step Guide:**

1. **Create Model Class** in `resource_prediction/models/implementations/`
   ```python
   from ..base import BasePredictor
   import pandas as pd
   import numpy as np
   
   class MyNewModel(BasePredictor):
       def __init__(self, param1: int = 100, param2: float = 0.1, random_state: int = 42, **kwargs):
           self.param1 = param1
           self.param2 = param2
           self.random_state = random_state
           # Your model initialization here
           
       def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
           # Implement training logic
           pass
           
       def predict(self, X: pd.DataFrame) -> np.ndarray:
           # Implement prediction logic
           pass
   ```

2. **Register in MODEL_FAMILIES** in `resource_prediction/config.py`
   ```python
   MODEL_FAMILIES = {
       "my_new_model_regression": {
           "type": "regression", 
           "base_model": "my_new_model",
           "class": _import_model_class("resource_prediction.models", "MyNewModel"),
       },
       # ... other models
   }
   ```

3. **Define Hyperparameter Configuration** in `config.py`
   ```python
   HYPERPARAMETER_CONFIGS = {
       "my_new_model_regression": {
           "use_quant_feats": {"choices": [True, False], "default": True},
           "param1": {"min": 50, "max": 200, "type": "int", "default": 100},
           "param2": {"min": 0.01, "max": 0.3, "type": "float", "log": True, "default": 0.1},
           # Only include parameters your model actually uses!
       }
   }
   ```

**That's it!** The dynamic import and instantiation system automatically handles the rest:

- ✅ **No manual imports needed** - Dynamic imports handle model loading automatically
- ✅ **No hardcoded logic** - Models are instantiated dynamically from MODEL_FAMILIES
- ✅ **Automatic registration** - Hyperparameter search discovers models automatically
- ✅ **Clean separation** - Each model only defines its own parameters

**Key Benefits of This Architecture:**
- ✅ **Simplified workflow** - Only 3 steps required to add new models
- ✅ **No parameter filtering** - Models accept parameters directly from hyperparameter search
- ✅ **Consistent interface** - Same model class used in search and evaluation  
- ✅ **Clean implementation** - Models handle their own parameter validation
- ✅ **Dynamic instantiation** - Evaluation uses `metadata['class']` for automatic model creation
- ✅ **No hardcoded logic** - Adding models doesn't require updating evaluation code

**Model-Specific Parameters**: Only include parameters that your specific model actually uses:
- Classification models automatically get `n_bins`, `strategy` from the classification config
- Regression models only need parameters relevant to their algorithm
- No need to handle `alpha` unless your model does quantile prediction
- Use `**kwargs` in your `__init__` to gracefully handle unexpected parameters

### Customizing Business Logic

- Modify the business scoring function in `Trainer` class
- Adjust the 5:1 penalty ratio for under vs over-allocation  
- Add new metrics (SLA compliance, cost thresholds, etc.)

### Interactive Web Application

The project includes a unified Streamlit web application for model exploration:

```console
# Launch the main application (only one needed)
streamlit run app/app.py
```

The application features:
- **Model Selection**: Radio button interface to choose between 4 different models:
  - Classification
  - Quantile Ensemble (3 variants: Balanced, Tiny Under-allocation, Small Waste)
- **Interactive Prediction**: Real-time memory prediction with simulation data
- **Visualization**: Live charts showing prediction behavior over time
- **Simulation Mode**: Automatic batch processing with configurable delay
- **Model-Specific Interfaces**: Each model type has its own optimized interface

The app dynamically loads the appropriate model and configuration based on user selection, with helper modules in the subdirectories (`app/qe/`, `app/classification/`) providing model-specific functionality.
