to be done

## Imagined "how to use" workflow:
### 1. First Run
```console
# Run full pipeline for classification
poetry run python main.py --model-type classification
```
### 2. Subsequent Run (skipping preprocessing)
```console
# Now regression, but re-use preprocessed data from first run
poetry run python main.py --model-type regression --skip-preprocessing
```

## How to Scale
- e.g. add new model type 
  - add new file in ```models/```, update ```config.py``` with its hyperparameters and add as choice in ```main.py``` and a condition in ```trainer.py```