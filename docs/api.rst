API Reference
=============

The following sections document the key modules of the project.

Configuration
-------------

.. automodule:: resource_prediction.config
   :members:
   :undoc-members:
   :show-inheritance:

Model Architecture
------------------

The unified model architecture provides a consistent interface for all prediction models.

BasePredictor Interface
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: resource_prediction.models.base
   :members:
   :undoc-members:
   :show-inheritance:

DeployableModel Wrapper
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: resource_prediction.models.unified_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Model Implementations
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: resource_prediction.models.implementations.lightgbm_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.models.implementations.xgboost_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.models.implementations.sklearn_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.models.implementations.quantile_ensemble_variants
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.models.implementations.sizey_model
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing Pipeline
----------------------

Model-time preprocessing used by DeployableModel:

.. automodule:: resource_prediction.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Training-time data preprocessing and feature engineering:

.. automodule:: resource_prediction.data_processing.preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

Training
--------

.. automodule:: resource_prediction.training.hyperparameter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.training.trainer
   :members:
   :undoc-members:
   :show-inheritance:

Reporting
---------

.. automodule:: resource_prediction.reporting
   :members:
   :undoc-members:
   :show-inheritance:

Pareto Tools
------------

Utilities for generating and using a Pareto frontier of LGB+XGB Quantile Ensemble configurations:

.. automodule:: resource_prediction.pareto.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.pareto.plot
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.pareto.export_models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.pareto.cli
   :members:
   :undoc-members:
   :show-inheritance:

Web Application
---------------

Interactive Streamlit application for model exploration and prediction.

.. automodule:: app.app
   :members:
   :undoc-members:
   :show-inheritance:

Entry Point
-----------

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:
