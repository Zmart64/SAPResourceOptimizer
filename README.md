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
├── classification/            # Classification model app
└── initial_approach/          # Initial regression app
```

## How to Extend

### Adding New Models

The unified model architecture makes extending the system straightforward:

1. **Create model implementation** extending `BasePredictor`:
   ```python
   # resource_prediction/models/my_model.py
   from .base import BasePredictor
   
   class MyPredictor(BasePredictor):
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

3. **Add to `Config.MODEL_FAMILIES`** in `resource_prediction/config.py`:
   ```python
   MODEL_FAMILIES = {
       # Existing models...
       "my_model_regression": {"type": "regression", "base_model": "my_model"},
       "my_model_classification": {"type": "classification", "base_model": "my_model"},
   }
   ```

4. **Add hyperparameter search space** to `Config.get_search_space()` method in the same file:
   ```python
   if base_model == 'my_model':
       return {
           "param1": trial.suggest_float("param1", 0.1, 1.0),
           "param2": trial.suggest_int("param2", 10, 100),
           # Add your hyperparameters
       }
   ```

5. **Update command-line options** in `main.py` if needed (optional for most cases)

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
- **Model Selection**: Radio button interface to choose between 5 different models:
  - Initial Approach (Classification)
  - Classification
  - Quantile Ensemble (3 variants: Balanced, Tiny Under-allocation, Small Waste)
- **Interactive Prediction**: Real-time memory prediction with simulation data
- **Visualization**: Live charts showing prediction behavior over time
- **Simulation Mode**: Automatic batch processing with configurable delay
- **Model-Specific Interfaces**: Each model type has its own optimized interface

The app dynamically loads the appropriate model and configuration based on user selection, with helper modules in the subdirectories (`app/qe/`, `app/classification/`, `app/initial_approach/`) providing model-specific functionality.
