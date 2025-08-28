# Resource Prediction

Predict peak memory requirements for distributed build jobs using a
combination of feature engineering and machine-learning models.

This project implements an automated ML pipeline that optimizes memory allocation
for CI/CD build systems, minimizing both out-of-memory failures and resource waste
through advanced hyperparameter optimization and business-focused metrics.

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
# Install dependencies (including development tools)
poetry install --with dev

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

Full project documentation is available in the `docs/` directory. To build and view locally:

```console
# Build the documentation
make html

# Build and serve the documentation (auto-opens in browser)
make serve

# For development: clean, build, and serve
make dev
```

The documentation will be available at `http://localhost:8000` and should automatically open in your browser.

Alternatively:

```console
poetry run sphinx-build -b html docs docs/_build/html
poetry run python serve_docs.py
```

**Benefits of DeployableModel Architecture:**
-  **Unified Interface** - All models implement `BasePredictor` for consistency
-  **Integrated Preprocessing** - `ModelPreprocessor` handles feature engineering automatically
-  **Production Ready** - Complete serialization with metadata and preprocessing pipeline
-  **Extensible Design** - Easy to add new models following the same pattern
-  **Clean Architecture** - Consistent parameter handling throughout hyperparameter search and evaluation

## Documentation

See the full docs in `docs/`. Build with `make html` or use `make serve` to host locally.

## Imagined "how to use" workflow

### 1. Simple Training (Recommended)

```console
# Activate Poetry environment
poetry shell

# Train all models with default parameters (simple and fast)
python main.py --train-default

# Train only classification models
python main.py --train-default --task-type classification

# Train specific model families
python main.py --train-default --model-families xgboost_classification lightgbm_regression
```

### 2. Advanced Training with Hyperparameter Search

```console
# Run full pipeline with hyperparameter optimization for classification
python main.py --run-search --task-type classification

# Now regression, but re-use preprocessed data from first run  
python main.py --run-search --task-type regression --skip-preprocessing

# Quick training with defaults (equivalent to --train-default)
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
| `--train-default` | Train models with default parameters | Quick prototyping, baseline results |
| `--run-search` | Full hyperparameter optimization | Best performance, production models |
| `--run-search --use-defaults` | Train with default hyperparameters instead of search (equivalent to `--train-default`) | Quick baselines |
| `--evaluate-only` | Evaluate existing trained models | Testing, comparison |

### Common Filtering Options

| Option | Description | Example |
|--------|-------------|---------|
| `--task-type` | Filter by regression or classification | `--task-type regression` |
| `--model-families` | Train specific model families | `--model-families xgboost_regression lightgbm_classification` |
| `--skip-preprocessing` | Reuse preprocessed data | `--skip-preprocessing --train-default` |

### Quick Examples

```console
# Simple: Train all models with defaults
python main.py --train-default

# Focused: Train only XGBoost models  
python main.py --train-default --model-families xgboost_regression xgboost_classification

# Fast: Reuse preprocessing from previous run
python main.py --train-default --skip-preprocessing

# Advanced: Full hyperparameter search for specific models
python main.py --run-search --model-families lightgbm_regression
```

## Features

- **Multiple ML Approaches**: Regression and classification models for memory prediction
- **Flexible Training Options**: Simple `--train-default` mode or advanced `--run-search` with hyperparameter optimization
- **Automated Hyperparameter Tuning**: Optuna-based optimization with parallel execution
- **Business-Focused Metrics**: Cost-aware objective function balancing failures vs waste
- **Rich Feature Engineering**: Temporal, categorical, and rolling window features
- **Unified Model Architecture**: Standardized BasePredictor interface for all models
- **Interactive Web Applications**: Streamlit-based dashboards for model exploration
- **Production-Ready**: Poetry dependency management, comprehensive testing, and documentation

### Quantile Ensemble Defaults

Multiple Quantile Ensemble (QE) variants are implemented (e.g., LGB+XGB, GB+LGB, XGB+XGB, Cat+Cat). On hold‑out data, these ensembles perform very similarly. To keep default runs fast and focused, only a single QE variant is active by default:

- Default QE: `lgb_xgb_ensemble` (LightGBM + XGBoost)
- Other QE variants are treated as experimental by default

Run all QE variants with `--run-all-qe-models`, or select specific families with `--model-families` (e.g., `gb_lgb_ensemble xgb_xgb_ensemble`). This matches the default behavior in `main.py` (default runs standard families + the default QE; the flag adds all QE variants).

### Training Modes

**Simple Training (`--train-default`)**:
- Train models with default parameters (no hyperparameter search)
- 90% simpler than `--run-search --use-defaults`
- Perfect for quick prototyping and baseline results
 - Example: `python main.py --train-default --model-families xgboost_regression`

**Advanced Training (`--run-search`)**:
- Full hyperparameter optimization using Optuna
- Best model performance but slower execution
- Use `--use-defaults` to skip search and use default parameters
- Example: `python main.py --run-search --model-families xgboost_regression`

 

## Supported Models

**Regression Models** (predict exact memory values):
- Quantile Ensemble (LGB+XGB default; other variants optional)
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
├── __init__.py                    # Public API exports
├── base.py                        # BasePredictor interface
├── unified_wrapper.py             # DeployableModel for production
└── implementations/               # Specific model implementations
    ├── lightgbm_models.py
    ├── quantile_ensemble_variants.py
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

The architecture makes it simple to add new models. Both hyperparameter search and final evaluation use the same wrapper classes, eliminating parameter filtering complexity.

**Step-by-Step Guide:**

1. **Create Model Wrapper** in `resource_prediction/models/implementations/`
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

2. **Register Model** in `resource_prediction/models/__init__.py`
   ```python
   from .implementations.my_new_model import MyNewModel
   
   __all__ = [
       "BasePredictor",
       "MyNewModel",  # Add your model here
       # ... other models
   ]
   ```

3. **Configure Model Family** in `resource_prediction/config.py`
   ```python
   MODEL_FAMILIES = {
       "my_new_model_regression": {
           "type": "regression", 
           "base_model": "my_new_model",
           "class": MyNewModel  # Reference for dynamic instantiation
       },
       # ... other models
   }
   ```

4. **Define Hyperparameter Space** in `config.py`
   ```python
   HYPERPARAMETER_CONFIGS = {
       "my_new_model_regression": {  # Use the full family name
           "param1": {"min": 50, "max": 200, "type": "int", "default": 100},
           "param2": {"min": 0.01, "max": 0.3, "type": "float", "log": True, "default": 0.1},
           # Only include parameters your model actually uses!
       }
   }
   ```



**That's it!** The system now uses dynamic model instantiation - no hardcoded model creation needed in evaluation!

**Key Benefits of This Architecture:**
-  **No Parameter Filtering** - Models accept parameters directly from hyperparameter search
-  **Consistent Interface** - Same model class used in search and evaluation  
-  **Clean Implementation** - Models handle their own parameter validation
-  **Dynamic Instantiation** - Evaluation uses `metadata['class']` for automatic model creation
-  **No Hardcoded Logic** - Adding models doesn't require updating evaluation code

**Model-Specific Parameters**: Only include parameters that your specific model actually uses:
- Classification models automatically get `n_bins`, `strategy` from the `classification_common` config
- Regression models only need parameters relevant to their algorithm
- No need to handle `alpha` unless your model does quantile prediction
- Use `**kwargs` in your `__init__` to gracefully handle unexpected parameters

### Customizing Business Logic

- Modify the business scoring function in `Trainer` class
- Adjust the 5:1 penalty ratio for under vs over-allocation  
- Add new metrics (SLA compliance, cost thresholds, etc.)

### Interactive Web Application

The project includes a unified Streamlit web application under the `app/` folder for model exploration:

```console
# Launch the main application
streamlit run app/app.py
```

The app structure includes:
- `app/app.py`: main Streamlit entrypoint and sidebar model registry
- `app/utils.py`: UI setup and simulation loop helpers
- `app/data_loader.py`: simulation data loading utilities

Available models are defined in the `available_models` dict in `app/app.py`. To add a new model, include its display name and the path to its serialized .pkl file in that dict.

### Pareto Tools

Utilities for analyzing and exporting Pareto‑optimal QE configurations live in `resource_prediction/pareto/` with a small CLI:

```bash
# Analyze frontier, plot focused chart, and export key models
python -m resource_prediction.pareto.cli all

# Or run individually
python -m resource_prediction.pareto.cli analyze
python -m resource_prediction.pareto.cli plot
python -m resource_prediction.pareto.cli export
```

Inputs/outputs are under `artifacts/pareto/` (frontier CSVs, focused plot, exported models). The app includes entries for the three exported presets (Balanced, Low Waste, Low Underallocation).
