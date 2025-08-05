# Resource Prediction

Predict peak memory requirements for distributed build jobs using a
combination of feature engineering and machine-learning models.

Full project documentation is available in the `docs/` directory. To
generate the HTML site locally, install the project with its development
dependencies (which include Sphinx) and then run the build:

```console
pip install -e .[dev]
sphinx-build -b html docs docs/_build
```

## Imagined "how to use" workflow

### 1. First run

```console
# Run full pipeline for classification
poetry run python main.py --model-type classification --run-search
```

### 2. Subsequent run (skipping preprocessing)

```console
# Now regression, but re-use preprocessed data from first run
poetry run python main.py --model-type regression --skip-preprocessing --run-search
```

## How to scale

- Add a new model type
  - create a file in `models/`, update `config.py` with its hyperparameters
    and add it as a choice in `main.py` and `trainer.py`
