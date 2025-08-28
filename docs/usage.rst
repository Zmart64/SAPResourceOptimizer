Usage
=====

This guide explains how to set up the project with Poetry and use the machine learning pipeline for resource prediction.

Prerequisites and Setup
------------------------

**Installing Poetry**

Poetry is used for dependency management and virtual environment handling. If you don't have Poetry installed:

.. code-block:: console

   # Install Poetry (macOS/Linux)
   curl -sSL https://install.python-poetry.org | python3 -

   # Or via pip
   pip install poetry

**Project Setup**

1. Clone or navigate to the project directory
2. Install dependencies using Poetry:

.. code-block:: console

   # Install all dependencies (including development tools like Sphinx)
   poetry install --with dev

3. Activate the Poetry shell to use the virtual environment:

.. code-block:: console

   # Activate the Poetry virtual environment
   poetry shell

   # Verify Python version in the activated environment
   python --version

**Building Documentation**

This project uses Sphinx for documentation generation. After installing dependencies, you can build and view the documentation locally:

.. code-block:: console

   # Build HTML documentation
   make html

   # Build and serve documentation (opens browser automatically)
   make serve

   # Development workflow: clean, build, and serve
   make dev

   # Clean build directory
   make clean

The documentation will be served at ``http://localhost:8000`` and should automatically open in your browser. Press ``Ctrl+C`` to stop the documentation server.

Training Pipeline Overview
--------------------------

The resource prediction pipeline is designed to predict peak memory requirements for distributed build jobs. The system uses machine learning to analyze build telemetry data and optimize memory allocation decisions.

**Pipeline Components:**

1. **Data Preprocessing**: Feature engineering on build telemetry data
2. **Hyperparameter Optimization**: Automated search using Optuna
3. **Model Training**: Multiple ML algorithms (regression and classification)
4. **Evaluation**: Business-focused metrics and final model selection

**DeployableModel Architecture:**

The project uses a sophisticated deployment architecture that separates concerns while providing a unified interface:

- **BasePredictor Interface**: All models implement this abstract base class ensuring consistent ``fit()`` and ``predict()`` methods  
- **DeployableModel Wrapper**: Production-ready wrapper that combines model + preprocessing + metadata
- **ModelPreprocessor Pipeline**: Sklearn-style preprocessing with ``fit()`` and ``transform()`` methods
- **Separation of Concerns**: Model logic, preprocessing, and deployment are cleanly separated
- **Extensible Design**: New models integrate seamlessly by implementing ``BasePredictor``

.. code-block:: python

   # Import models using the new architecture
   from resource_prediction.models import BasePredictor, DeployableModel
   from resource_prediction.preprocessing import ModelPreprocessor
   
   # Load a production-ready model with integrated preprocessing
   model = DeployableModel.load("artifacts/trained_models/lightgbm_regression.pkl")
   predictions = model.predict(raw_dataframe)  # Preprocessing happens automatically

First run
---------

Execute the full pipeline for a specific task type:

.. code-block:: console

   python main.py --task-type classification --run-search

Subsequent runs
---------------

To reuse previously processed data and run a different task type:

.. code-block:: console

   python main.py --task-type regression --skip-preprocessing --run-search

Command line options
--------------------

``--run-search``
    Perform Optuna hyper-parameter optimisation and final evaluation.
``--skip-preprocessing``
    Assume preprocessed data exists and skip feature engineering.
``--preprocess-only``
    Run the preprocessing step and exit without optimisation.
``--evaluate-all-archs``
    Evaluate the best model from each architecture and generate a
    comparison chart.
``--task-type {regression,classification}``
    Restrict the pipeline to a single task.

Training Pipeline Details
-------------------------

**Machine Learning Approaches**

The pipeline supports two main approaches to memory prediction:

1. **Regression Models**: Predict exact memory values in GB
   
   - **Quantile Ensemble**: Combines GradientBoosting and XGBoost quantile regressors for conservative predictions
   - **XGBoost Regression**: Direct memory prediction with L1 regularization
   - **LightGBM Regression**: Fast gradient boosting with memory efficiency

2. **Classification Models**: Predict memory bins/categories
   
   - **XGBoost Classifier**: Multi-class prediction with gradient boosting
   - **LightGBM Classifier**: Fast gradient boosting with leaf-wise tree growth
   - **Random Forest Classifier**: Ensemble classification approach
   - **Logistic Regression**: Linear baseline for multi-class prediction

**Hyperparameter Optimization**

The pipeline uses `Optuna <https://optuna.org/>`_ for automated hyperparameter tuning:

- **Search Space**: Each model family has carefully defined search spaces
- **Cross-Validation**: Time-series splits to respect temporal dependencies
- **Parallel Execution**: Multiple trials run concurrently for efficiency
- **Pruning**: Early stopping for unpromising trials

**Classification-Specific Hyperparameters**

For classification models, the pipeline treats the discretization of continuous memory values as hyperparameters:

- **Number of Bins** (``n_bins``): Optimized range of 3-15 bins for memory value discretization
- **Binning Strategy** (``strategy``): Choice between three approaches:
  
  - ``uniform``: Equal-width bins across the memory range
  - ``quantile``: Bins based on quantiles of the data distribution
  - ``kmeans``: Bins determined by K-means clustering centers

This approach allows the system to find the optimal granularity for converting the continuous memory prediction problem into a multi-class classification task, balancing prediction precision with model complexity.

**Business Function**

The optimization objective combines two critical business metrics:

.. math::

   \text{Business Score} = 5 \times \text{Under-allocation \%} + \text{Over-allocation \%}

Where:

- **Under-allocation %**: Percentage of jobs that receive insufficient memory (causes failures)
- **Over-allocation %**: Percentage of total memory wasted through over-provisioning

This scoring function reflects that under-allocation is 5x more costly than over-allocation, as failed builds have significant business impact (developer time, CI/CD delays, resource waste).

**Feature Engineering**

The preprocessor creates rich features from raw build telemetry:

- **Temporal Features**: Year, month, day-of-week, hour, week-of-year
- **Categorical Features**: Location, component, build type, architecture
- **Derived Features**: Branch prefixes, target counts, parallelism indicators
- **Rolling Window Features**: Historical memory usage patterns
- **Quantitative Features**: Build load, target intensity, debug multipliers

**Evaluation Metrics**

Final model performance is assessed using:

- **Cross-validation Score**: Business score during hyperparameter search
- **Hold-out Metrics**: Final evaluation on unseen test data
- **Under-allocation Rate**: Critical failure prevention metric
- **Memory Waste**: Resource efficiency measurement
- **Total Over-allocation**: Infrastructure cost implications

Extending the project
---------------------

New model families can be added under :mod:`resource_prediction.models`
and registered in :mod:`resource_prediction.config`.  The
:class:`resource_prediction.training.trainer.Trainer` class will pick up
additional families automatically when the configuration is updated.

**Adding New Models**

The system is designed for easy extensibility with new model types. Here's a comprehensive guide to adding a new model:

**Step 1: Implement the BasePredictor Interface**

Create your model class by extending the :class:`~resource_prediction.models.base.BasePredictor` abstract class. This ensures consistency across the system:

.. code-block:: python

   # resource_prediction/models/my_new_model.py
   from .base import BasePredictor
   import pandas as pd
   import numpy as np

   class MyNewModel(BasePredictor):
       """Example new model implementation."""
       
       def __init__(self, param1=100, param2=0.1, task_type='regression', **kwargs):
           self.param1 = param1
           self.param2 = param2
           self.task_type = task_type
           self.is_fitted = False
           
       def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
           """Train the model on the provided data."""
           # Your training logic here
           # Example: self.model = SomeLibrary.Model(param1=self.param1)
           # self.model.fit(X.values, y.values)
           self.is_fitted = True
           return self
           
       def predict(self, X: pd.DataFrame) -> np.ndarray:
           """Make predictions on new data."""
           if not self.is_fitted:
               raise ValueError("Model must be fitted before prediction")
           # Your prediction logic here
           # return self.model.predict(X.values)
           return np.random.random(len(X))  # Placeholder
           
       def get_params(self) -> dict:
           """Return model parameters for hyperparameter optimization."""
           return {
               'param1': self.param1,
               'param2': self.param2,
               'task_type': self.task_type
           }

**Step 2: Register the Model**

Add your model to the module registry:

.. code-block:: python

   # resource_prediction/models/__init__.py
   from .my_new_model import MyNewModel

   __all__ = [
       "BasePredictor",
       "DeployableModel", 
       "QuantileEnsemblePredictor",
       "MyNewModel"  # Add your new model here
   ]

**Step 3: Configure Model Families**

Add your model to the configuration with appropriate family names:

.. code-block:: python

   # resource_prediction/config.py - Add to MODEL_FAMILIES dictionary
   MODEL_FAMILIES = {
       # ... existing models ...
       "my_new_model_regression": {"type": "regression", "base_model": "my_new_model"},
       "my_new_model_classification": {"type": "classification", "base_model": "my_new_model"},
   }

**Step 4: Define Hyperparameter Search Space**

Add search space configuration to enable hyperparameter optimization:

.. code-block:: python

   # resource_prediction/config.py - Add to get_search_space() method
   def get_search_space(self, family_name: str, trial, use_quant: bool = True) -> dict:
       # ... existing code ...
       
       if base_model == 'my_new_model':
           model_params = {
               "param1": trial.suggest_int("param1", 50, 200),
               "param2": trial.suggest_float("param2", 0.01, 1.0, log=True),
           }
           return {**common_params, **model_params} if task_type == 'classification' else model_params

**Step 5: No Trainer Changes Needed**

The trainer dynamically instantiates models using each family’s configured class reference in
``Config.MODEL_FAMILIES``. No changes to trainer code are required.

**Step 6: Test Your Integration**

Verify your model works with the pipeline:

.. code-block:: console

   # Test with hyperparameter search
   poetry run python main.py --run-search --task-type regression

   # Test specific model family
   poetry run python main.py --run-search --model-family my_new_model_regression

**What You Get Automatically:**

Once integrated, your model automatically receives:

- ✅ **Hyperparameter Optimization**: Optuna-based search with your defined parameter space
- ✅ **Cross-Validation**: Automatic k-fold evaluation with business metrics
- ✅ **DeployableModel Wrapper**: Production-ready packaging with preprocessing
- ✅ **Model Serialization**: Automatic saving with metadata and version info
- ✅ **Evaluation Metrics**: Business scoring, accuracy, and comparison charts
- ✅ **Integration Testing**: Compatibility with simulation and web applications

**Advanced Customization:**

For more sophisticated models, you can:

- **Custom Preprocessing**: Override preprocessing in your model's predict method
- **Ensemble Integration**: Combine with existing models in the QuantileEnsemble
- **Custom Metrics**: Add model-specific evaluation metrics
- **Parameter Dependencies**: Create conditional search spaces based on other parameters

**System Architecture Overview**

The project follows a clean three-layer architecture:

**1. Model Layer** (``resource_prediction/models/``)
   - **BasePredictor**: Abstract interface ensuring consistent model behavior
   - **Concrete Models**: XGBoost, LightGBM, Random Forest, Logistic Regression, Quantile Ensemble
   - **DeployableModel**: Production wrapper with integrated preprocessing

**2. Training Layer** (``resource_prediction/training/``)
   - **HyperparameterSearcher**: Optuna-based optimization with business metrics
   - **Trainer**: Model evaluation, cross-validation, and champion selection
   - **Business Scoring**: Domain-specific metrics (5:1 penalty for under-allocation)

**3. Application Layer** (``app/``)
   - **Streamlit Interface**: Interactive model comparison and simulation
   - **Data Pipeline**: Automated preprocessing and feature engineering
   - **Production Integration**: Direct model loading and prediction serving

**Key Benefits:**

- **Consistency**: All models implement the same interface via BasePredictor
- **Flexibility**: Easy to swap models without changing application code  
- **Production-Ready**: DeployableModel ensures models include all preprocessing
- **Extensible**: Adding new models requires minimal code changes

**Data Flow:**

1. **Raw Data** → **ModelPreprocessor** → **Engineered Features**
2. **Features** → **BasePredictor.fit()** → **Trained Model**
3. **Trained Model** → **DeployableModel** → **Production Artifact**
4. **New Data** → **DeployableModel.predict()** → **Memory Allocations**

This architecture ensures that models are interchangeable, preprocessing is consistent, and the system remains maintainable as new model types are added.

**Customizing Business Logic**

The business scoring function can be modified in the ``Trainer`` class:

- Adjust the 5:1 penalty ratio for under vs over-allocation
- Add new metrics (e.g., SLA compliance, cost thresholds)
- Implement domain-specific constraints

Example Workflows
-----------------

**Complete Pipeline Run**

.. code-block:: console

   # Full pipeline: preprocessing + hyperparameter search + evaluation
   python main.py --run-search

**Development Workflow**

.. code-block:: console

   # 1. Initial data preprocessing
   python main.py --preprocess-only
   
   # 2. Quick classification experiment
   python main.py --task-type classification --skip-preprocessing --run-search
   
   # 3. Compare with regression approach
   python main.py --task-type regression --skip-preprocessing --run-search
   
   # 4. Full architecture comparison with plots
   python main.py --skip-preprocessing --run-search --evaluate-all-archs

**Production Deployment**

.. code-block:: console

   # Focus on regression models for production use
   python main.py --task-type regression --run-search --evaluate-all-archs

This will generate performance comparisons and help select the best model for deployment.

Interactive Web Application
---------------------------

The project includes a unified Streamlit application for interactive model exploration and prediction.

.. code-block:: console

   # Launch the main application
   streamlit run app/app.py

The app provides:

- **Model selection** in the sidebar via a curated registry of models
- **Interactive predictions** over simulation data with live charts
- **Adjustable speed** of the simulation loop and classification overlays

Application layout:

.. code-block:: text

   app/
   ├── app.py          # Main Streamlit application
   ├── utils.py        # UI setup and simulation helpers
   └── data_loader.py  # Simulation data loading utilities

Models are wired in via the ``available_models`` dictionary in ``app/app.py`` that maps display names to
paths of serialized models (.pkl) under ``artifacts/``. The app uses ``DeployableModel``’s ``load()`` and
``predict()`` methods, so preprocessing is applied automatically.

**Trained Model Artifacts**

Pre-trained models are organized in the ``artifacts/trained_models/`` directory:

.. code-block:: text

   artifacts/
   ├── trained_models/
   │   ├── lightgbm_classification.pkl
   │   ├── xgboost_classification.pkl
   │   ├── lightgbm_regression.pkl
   │   └── xgboost_regression.pkl
   └── pareto/models/
       ├── qe_balanced.pkl
       ├── qe_low_waste.pkl
       └── qe_low_underallocation.pkl

These artifacts demonstrate using trained models in production-like scenarios and enable exploration without running the full training pipeline.

Quantile Ensemble Defaults
--------------------------

This repository supports multiple Quantile Ensemble (QE) variants (e.g., LGB+XGB, GB+LGB, XGB+XGB, etc.). In practice, hold‑out performance among these ensembles is very similar. To keep runs fast and the code path simple, only a single, well‑performing QE is active by default:

- Default QE: ``lgb_xgb_ensemble`` (LightGBM + XGBoost)
- Other QE variants are considered experimental and are excluded by default

Include all QE variants by either:

- Passing ``--run-all-qe-models``
- Or explicitly listing the families with ``--model-families`` (e.g., ``gb_lgb_ensemble xgb_xgb_ensemble``)

This mirrors ``main.py`` behavior: without filters, all non‑experimental families plus the default QE run; with the flag, all QEs are added.

Pareto Tools
------------

Utilities for analyzing and exporting Pareto‑optimal QE configurations are available under
``resource_prediction/pareto/`` with a simple CLI:

.. code-block:: console

   # Analyze frontier, plot a focused chart, and export key models
   python -m resource_prediction.pareto.cli all

   # Or run individually
   python -m resource_prediction.pareto.cli analyze
   python -m resource_prediction.pareto.cli plot
   python -m resource_prediction.pareto.cli export

Artifacts are written under ``artifacts/pareto/`` (frontier CSVs, focused plot, exported models).

Prerequisites and considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``export`` requires a saved deployable champion for ``lgb_xgb_ensemble`` at
  ``artifacts/trained_models/lgb_xgb_ensemble.pkl``. Create it by running the main pipeline with
  ``--save-models`` (for example: ``python main.py --run-search --task-type regression --save-models``).
- ``sweep`` (frontier generation) needs preprocessed data. It will reuse tuned hyperparameters for
  ``lgb_xgb_ensemble`` from ``artifacts/experiments/regression_results.csv`` if present; otherwise it
  falls back to safe defaults.
- ``analyze`` and ``plot`` expect existing frontier CSVs under ``artifacts/pareto/results/``.
  Run ``pareto sweep`` first to (re)generate them.
