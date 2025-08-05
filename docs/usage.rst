Usage
=====

The project is orchestrated through :mod:`main` which exposes a command
line interface.  Typical workflows involve preprocessing the data and
running hyper-parameter searches.

First run
---------

Execute the full pipeline for a specific task type:

.. code-block:: console

   poetry run python main.py --model-type classification --run-search

Subsequent runs
---------------

To reuse previously processed data and run a different task type:

.. code-block:: console

   poetry run python main.py --model-type regression --skip-preprocessing --run-search

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

Extending the project
---------------------

New model families can be added under :mod:`resource_prediction.models`
and registered in :mod:`resource_prediction.config`.  The
:class:`resource_prediction.training.trainer.Trainer` class will pick up
additional families automatically when the configuration is updated.
