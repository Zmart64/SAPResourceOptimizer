# Build Job Memory Predictor

This project provides a machine learning pipeline to predict the peak memory usage (max_rss) of software build jobs. It uses a data-driven approach to select the best model and binning strategy, then trains it and provides a real-time simulation dashboard.

## Project Workflow

The project follows a three-stage workflow:

1. **Bayesian Optimization (Prerequisite):** A preceding process runs a multi-model Bayesian optimization to find the best-performing model (e.g., XGBoost, CatBoost) and memory binning strategy. This step generates a crucial `best_strategy_summary.csv` file.

2. **Train Best Performer:** The main training script reads the optimal configuration from `best_strategy_summary.csv`, trains the specified model on the first 80% of the historical data, and saves the final model and the remaining 20% of the data for simulation.

3. **Real-time Simulation:** A Streamlit application loads the saved model and the 20% holdout data to simulate and visualize the model's performance on new, incoming jobs.

## Getting Started

### Prerequisites

- Python 3.8+
- pip for package installation

### 1. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Step 1: Provide Strategy Configuration

Before training, you must have the output from the optimization phase.

Place the `best_strategy_summary.csv` file in the root directory of the project. This file is required to tell the training script which model to build.

### 3. Step 2: Train the Model

With the configuration file in place, run the main training script:

```bash
python train_best_performer.py
```

This script will:

- Read `best_strategy_summary.csv` to get the model configuration.
- Train the specified model on 80% of `build-data-sorted.csv`.
- Save the trained model, reports, and the 20% holdout data to a new directory (e.g., `output_plots_best_performer/random_forest_final_tuned/`).

### 4. Step 3: Run the Simulation

Once training is complete, launch the Streamlit dashboard:

```bash
streamlit run app.py
```

This will open a browser tab simulating the model's performance on the 20% holdout data.

## File Structure

```text
.
├── output_plots_best_performer/  # Created by the training script
│   └── <model_name>_final_tuned/
│       ├── final_model.pkl
│       └── simulation_holdout_data.csv
│       └── ... (reports)
├── streamlit_app.py              # The Streamlit dashboard
├── train_best_performer.py       # Main training script
├── build-data-sorted.csv         # The full dataset
├── best_strategy_summary.csv     # INPUT: Configuration for the trainer
├── requirements.txt
└── README.md
```
