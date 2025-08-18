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

.. automodule:: resource_prediction.models.quantile_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing Pipeline
----------------------

.. automodule:: resource_prediction.preprocessing
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

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance:

Entry Point
-----------

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:
