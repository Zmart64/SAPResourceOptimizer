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

### Using Models (New Unified Interface)

The project now provides a unified interface for all model types, eliminating the need for model-specific preprocessing:

```python
from resource_prediction.models import load_any_model

# Load any model type with consistent interface
model = load_any_model("artifacts/trained_models/lightgbm_classification.pkl")

# Make predictions on raw data (no preprocessing needed!)
predictions = model.predict(raw_dataframe)

# Get model information  
model_info = model.get_model_info()
print(f"Model: {model_info['model_type']} ({model_info['task_type']})")
```

**Benefits of Unified Interface:**
- 🔄 **Single interface** for all model types (classification, regression, QE)
- 🧠 **Automatic preprocessing** - no manual feature engineering required
- 🏗️ **Easy integration** - works seamlessly with existing and new models
- 📦 **Backward compatible** - legacy models continue to work

See [docs/unified_models.md](docs/unified_models.md) for comprehensive documentation.

### Documentation

Full project documentation is available in the `docs/` directory. To build and view the documentation locally:

```console
# Build the documentation
make html

# Build and serve the documentation (auto-opens in browser)
make serve

# For development: clean, build, and serve
make dev
```

The documentation will be available at `http://localhost:8000` and should automatically open in your browser.

**Alternative manual commands:**
```console
# If you prefer manual commands
poetry run sphinx-build -b html docs docs/_build/html
poetry run python serve_docs.py
```

## Imagined "how to use" workflow

### 1. First run

```console
# Activate Poetry environment
poetry shell

# Run full pipeline for classification
python main.py --model-type classification --run-search
```

### 2. Subsequent run (skipping preprocessing)

```console
# Now regression, but re-use preprocessed data from first run
python main.py --model-type regression --skip-preprocessing --run-search
```

## Features

- **Multiple ML Approaches**: Regression and classification models for memory prediction
- **Automated Hyperparameter Tuning**: Optuna-based optimization with parallel execution
- **Business-Focused Metrics**: Cost-aware objective function balancing failures vs waste
- **Rich Feature Engineering**: Temporal, categorical, and rolling window features
- **Unified Model Architecture**: Standardized BasePredictor interface for all models
- **Interactive Web Applications**: Streamlit-based dashboards for model exploration
- **Production-Ready**: Poetry dependency management, comprehensive testing, and documentation

## Supported Models

**Regression Models** (predict exact memory values):
- Quantile Ensemble (GradientBoosting + XGBoost)
- XGBoost Regression
- Random Forest Regression

**Classification Models** (predict memory bins):
- XGBoost Classifier
- LightGBM Classifier  
- CatBoost Classifier
- Random Forest Classifier
- Logistic Regression

## Architecture

The pipeline implements a business-focused optimization objective:

**Business Score = 5 × Under-allocation% + Over-allocation%**

This reflects that memory under-allocation (causing build failures) is 5x more 
costly than over-allocation (wasting resources).

### Unified Model Architecture

The project uses a standardized model interface (`BasePredictor`) that ensures consistency across all implementations:

```python
# All models implement the same interface
from resource_prediction.models import BasePredictor, QEPredictor, QuantileEnsemblePredictor

# Unified import structure
model = QEPredictor()  # Backward compatible alias
model = QuantileEnsemblePredictor()  # New explicit name
```

**Key architectural principles:**

- **Single Source of Truth**: All model definitions consolidated in `resource_prediction/models/`
- **Clean Separation**: Model definitions separate from trained artifacts (`artifacts/trained_models/`)
- **Backward Compatibility**: Existing code continues to work without changes
- **Extensible Interface**: `BasePredictor` makes adding new models straightforward

### Directory Structure

```
resource_prediction/models/     # Python model definitions
├── base.py                     # BasePredictor interface
├── quantile_ensemble.py        # Unified QE implementation
└── __init__.py                 # Model registry

artifacts/trained_models/       # Saved model files (.pkl)
├── app/                        # Models for web applications
└── resource_prediction/        # Models from training pipeline

app/                           # Interactive web applications
├── app.py                     # Main Streamlit dashboard
├── qe/                        # Quantile ensemble app
└── classification/            # Classification model app
```

## How to Extend

### Adding New Models

The unified model architecture makes extending the system straightforward. Here's the complete process using an example `MyPredictor` model:

1. **Create model implementation** extending `BasePredictor`:
   ```python
   # resource_prediction/models/my_model.py
   from .base import BasePredictor
   
   class MyPredictor(BasePredictor):
       def __init__(self, param1=0.5, param2=50, **kwargs):
           self.param1 = param1
           self.param2 = param2
           
       def fit(self, X, y, **fit_params):
           # Implementation
           pass
       
       def predict(self, X):
           # Implementation
           pass
   ```

2. **Register in model registry** (`resource_prediction/models/__init__.py`):
   ```python
   from .my_model import MyPredictor
   
   __all__ = [
       "BasePredictor",
       "QEPredictor",
       "QuantileEnsemblePredictor",
       "MyPredictor"  # Add your new model
   ]
   ```

3. **Choose a base_model identifier** and add to `Config.MODEL_FAMILIES` in `resource_prediction/config.py`:
   ```python
   MODEL_FAMILIES = {
       # Existing models...
       "my_model_regression": {"type": "regression", "base_model": "my_custom_model"},
       "my_model_classification": {"type": "classification", "base_model": "my_custom_model"},
   }
   # Note: "my_custom_model" is your chosen identifier string - it can be anything
   # This string will be used to connect your config to your model instantiation
   ```

4. **Add hyperparameter search space** to `Config.get_search_space()` method using your identifier:
   ```python
   if base_model == 'my_custom_model':  # Must match your identifier from step 3
       return {
           "param1": trial.suggest_float("param1", 0.1, 1.0),
           "param2": trial.suggest_int("param2", 10, 100),
           "use_quant_feats": trial.suggest_categorical("use_quant_feats", [True, False]),
           # Add your hyperparameters
       }
   ```

5. **Add model instantiation** in `resource_prediction/training/hyperparameter.py` in the `_objective` method:
   ```python
   # In the regression section (around line 129):
   if base_model == 'my_custom_model':  # Must match your identifier
       model = MyPredictor(
           param1=params["param1"],
           param2=params["param2"],
           random_state=self.config.RANDOM_STATE
       )
   
   # And/or in the classification section (around line 153):
   elif base_model == 'my_custom_model':  # Must match your identifier  
       model = MyPredictor(
           param1=params["param1"],
           param2=params["param2"],
           random_state=self.config.RANDOM_STATE
       )
   ```

**Key Points:**
- The `base_model` string (e.g., `"my_custom_model"`) is just an identifier you choose
- This string must be consistent across `config.py`, `get_search_space()`, and `hyperparameter.py`  
- The training system uses explicit if-elif statements to map your string to your model class
- Your model class name (e.g., `MyPredictor`) can be different from your identifier string

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
