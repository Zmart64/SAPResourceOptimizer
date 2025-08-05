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

First run
---------

Execute the full pipeline for a specific task type:

.. code-block:: console

   python main.py --model-type classification --run-search

Subsequent runs
---------------

To reuse previously processed data and run a different task type:

.. code-block:: console

   python main.py --model-type regression --skip-preprocessing --run-search

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
   - **Random Forest Regression**: Ensemble method for robust predictions

2. **Classification Models**: Predict memory bins/categories
   
   - **XGBoost Classifier**: Multi-class prediction with gradient boosting
   - **LightGBM Classifier**: Fast gradient boosting with leaf-wise tree growth
   - **CatBoost Classifier**: Handles categorical features automatically
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

1. Create model implementation in ``resource_prediction/models/``
2. Add hyperparameter search space to ``Config.get_search_space()``
3. Register in ``Config.MODEL_FAMILIES`` dictionary
4. Update command-line options if needed

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
