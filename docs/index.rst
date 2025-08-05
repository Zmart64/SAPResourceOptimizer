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
- Poetry-based dependency management
- Comprehensive evaluation and reporting

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   data
   api
