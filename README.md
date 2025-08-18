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

Full project documentation is available in the `docs/` directory. To build and view the documentation locally:

```console
# Build the documentation
make html

# Build and serve the documentation (auto-opens in browser)
make serve

# For development: clean, build, and serve
make dev
```
```

**Benefits of DeployableModel Architecture:**
- 🔄 **Unified Interface** - All models implement `BasePredictor` for consistency
- 🧠 **Integrated Preprocessing** - `ModelPreprocessor` handles feature engineering automatically
- 🏗️ **Production Ready** - Complete serialization with metadata and preprocessing pipeline
- 📦 **Extensible Design** - Easy to add new models following the same pattern
- � **Type Safety** - Clear separation between model logic and deployment wrapper

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

The project uses a clean model architecture:

- **BasePredictor Interface**: All models implement consistent `fit()` and `predict()` methods
- **DeployableModel Wrapper**: Production-ready wrapper with integrated preprocessing
- **ModelPreprocessor**: Handles feature engineering automatically

## How to Extend

### Adding New Models

To add a new model type:

1. **Implement BasePredictor interface** in `resource_prediction/models/`
2. **Add to model registry** in `resource_prediction/models/__init__.py`  
3. **Configure model family** in `resource_prediction/config.py`
4. **Define search space** in `config.py`
5. **Add trainer integration** in `resource_prediction/training/trainer.py`

**Example: Adding a new regression model**

```python
# In resource_prediction/config.py HYPERPARAMETER_CONFIGS
"my_regression_model": {
    "n_estimators": {"min": 50, "max": 200, "type": "int", "default": 100},
    "max_depth": {"min": 3, "max": 10, "type": "int", "default": 6},
    "learning_rate": {"min": 0.01, "max": 0.3, "type": "float", "log": True, "default": 0.1},
    # No alpha or use_quant_feats required - only include parameters your model actually needs!
}
```

**Model-Specific Parameters**: The hyperparameter system now uses model-specific configuration rather than forcing universal parameters. This means:
- New regression models don't need `alpha` or `use_quant_feats` unless they use quantile prediction
- Only include parameters that your specific model actually uses
- No more parameter conflicts when adding new model types

See the documentation for detailed examples.

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
