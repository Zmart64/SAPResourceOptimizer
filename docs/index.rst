Resource Prediction Documentation
=================================

This documentation provides an in-depth overview of the resource
prediction project, a machine learning pipeline designed to predict peak memory 
requirements for distributed build jobs. The system uses advanced feature 
engineering and automated hyperparameter optimization to minimize both 
under-allocation failures and memory waste.

**Key Features:**

- Automated hyperparameter search using Optuna
- Multiple ML algorithms (regression and classification approaches)
- Business-focused optimization metrics
- Rich feature engineering from build telemetry
- Unified model architecture with standardized interfaces
- Interactive Streamlit web applications for model exploration
- Poetry-based dependency management
- Comprehensive evaluation and reporting

**System Architecture:**

The project follows a clean three-layer architecture:

- **DeployableModel Layer**: Production wrapper with integrated preprocessing and metadata
- **BasePredictor Layer**: Abstract interface ensuring consistent model behavior  
- **Implementation Layer**: Concrete ML algorithms (XGBoost, LightGBM, Random Forest, Logistic Regression, Quantile Ensemble)

This architecture enables seamless integration of new models while maintaining production readiness and type safety.

Quick Start
-----------

To get started with this project:

1. **Install dependencies**: ``poetry install --with dev``
2. **Build documentation**: ``make html`` or ``make serve``
3. **Run training**: ``python main.py --train-default`` (fast) or ``python main.py --run-search`` (optimized)

For detailed setup instructions, see the :doc:`usage` guide.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   data
   cli
   api
