CLI Reference
=============

This page summarizes the command-line interface for the main pipeline entrypoint ``main.py``.

Overview
--------

Run from the project root with Poetry or your active environment:

.. code-block:: console

   # Examples
   python main.py --train-default
   python main.py --run-search --task-type regression
   python main.py --evaluate-only --evaluate-all-archs

Actions (choose one)
--------------------

Exactly one of the following actions is required:

``--run-search``
    Run full hyperparameter optimization and final evaluation.

``--train-default``
    Train models with default hyperparameters and evaluate them (a simpler alternative to ``--run-search --use-defaults``).

``--evaluate-only``
    Evaluate using parameters from champion result CSV files (skips hyperparameter search).

``--preprocess-only``
    Run the data preprocessing step and exit.

Common options
--------------

``--skip-preprocessing``
    Reuse existing processed data instead of recomputing it.

``--evaluate-all-archs``
    During final evaluation, evaluate the best model from all architectures and generate a comparison plot.

``--task-type {regression,classification}``
    Restrict the pipeline to a single task type. By default both run.

``--save-models``
    Save final evaluated champion model(s) as serialized ``.pkl`` files under ``artifacts/trained_models``.

``--model-families <names...>``
    Limit runs to specific model families (space-separated). To see available families, check ``Config.MODEL_FAMILIES`` in ``resource_prediction/config.py``.

``--use-defaults``
    With ``--run-search``, skip optimization and train with default hyperparameters (similar to ``--train-default``).

``--run-all-qe-models``
    Include all Quantile Ensemble (QE) variants in addition to the default QE. Without this flag, only the default QE (``lgb_xgb_ensemble``) is included because it performs best and other ensembles have similar performance.

Quantile Ensemble defaults
--------------------------

By default, runs include the standard families and a single, well-performing QE (``lgb_xgb_ensemble``). Other QE variants are considered experimental and excluded unless you pass ``--run-all-qe-models`` or explicitly list them via ``--model-families``.

Quick examples
--------------

.. code-block:: console

   # Fast baseline (no search)
   python main.py --train-default

   # Full search for classification
   python main.py --run-search --task-type classification

   # Reuse preprocessed data, focus on regression, evaluate all archs
   python main.py --run-search --task-type regression --skip-preprocessing --evaluate-all-archs

   # Run specific families only
   python main.py --run-search --model-families lightgbm_regression xgboost_classification

   # Include all QE variants
   python main.py --run-search --run-all-qe-models

