API Reference
=============

The following sections document the key modules of the project.

Configuration
-------------

.. automodule:: resource_prediction.config
   :members:
   :undoc-members:
   :show-inheritance:

Model Interface
---------------

The unified model architecture provides a consistent interface for all prediction models.

.. automodule:: resource_prediction.models.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: resource_prediction.models.quantile_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

Data Processing
---------------

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

Web Applications
----------------

Interactive Streamlit applications for model exploration and prediction.

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
