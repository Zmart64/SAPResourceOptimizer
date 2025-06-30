# Build Job Memory Predictor

This project provides a machine learning pipeline to predict the peak memory usage (`max_rss`) of software build jobs. It uses an XGBoost Classifier to categorize jobs into predefined memory "bins", allowing for more robust resource allocation. The final output is a real-time Streamlit dashboard that simulates and visualizes the model's performance on unseen data.

## Key Concepts

### The Approach: Classification via Binning

Instead of predicting an exact continuous memory value (regression), this project frames the problem as a classification task.

*   **Why?** Predicting a specific memory value is difficult and sensitive to noise. By grouping memory usage into bins, the model can focus on predicting the correct *range* of memory required, which is more practical for resource allocation (e.g., assigning a job to a machine with 64GB, 128GB, or 256GB of RAM).

*   **How?** The `max_rss` target variable is converted into discrete classes. The model then predicts which memory bin a job is most likely to fall into. The following bin edges (in Gigabytes) are used:
    ```
    [3.00e-02, 5.77e+01, 1.15e+02, 1.73e+02, 2.31e+02]
    ```
    > **Note:** These specific bin edges were identified as the top performers through a **Bayesian optimization process**, ensuring they are well-suited and optimized for this dataset.

### Data Split: Training vs. Simulation

To ensure a fair and realistic evaluation of the model, the project uses a strict chronological **80/20 data split**:

*   **Training Data (First 80%)**: The first 80% of the time-sorted data is used *exclusively* for all model development activities. This includes hyperparameter tuning (GridSearchCV) and cross-validation (`TimeSeriesSplit`). The model never sees the final 20% of the data during this phase.

*   **Simulation Data (Final 20%)**: The final 20% of the data is held out as a completely "unseen" dataset. This data is used *only* by the Streamlit application to simulate how the trained model would perform on new, incoming jobs in a real-world scenario.

This approach prevents data leakage and gives a true measure of the model's generalization performance.

## Getting Started

Follow these steps to set up the environment, train the model, and run the simulation.

### Prerequisites

*   Python 3.8+
*   `pip` for package installation

### 1. Installation

First, create a `requirements.txt` file in your project's root directory with the following content:

**`requirements.txt`**
```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
altair
tqdm
tqdm_joblib
```

Now, install all the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Training the Model

Run the main training script from your terminal. This script performs the 80/20 split, trains the model on the first 80%, and saves the necessary artifacts.
```bash
python model_trainer.py
```

This script will:

- Load the full dataset from `build-data-sorted.csv`.
- Perform the chronological 80/20 split.
- Train the XGBoost classifier on the training set using cross-validation and hyperparameter search.
- Save two crucial files into the `output_model_and_plots/` directory:
    - `xgb_classifier_model.pkl`: The trained model, feature list, and binning information.
    - `simulation_data.csv`: The 20% holdout data for the simulation.

### 3. Running the Real-Time Simulation

Once the training is complete and the artifacts are saved, launch the Streamlit dashboard:
```bash
streamlit run app.py
```

This will open a new tab in your web browser. The application loads the trained model and the `simulation_data.csv` to simulate and visualize the model's performance on the unseen data, providing metrics on allocation accuracy and waste.

## File Structure
```
.
├── output_model_and_plots/   # Created by the training script
│   ├── xgb_classifier_model.pkl
│   └── simulation_data.csv
├── app.py                    # The Streamlit dashboard application
├── model_trainer.py          # Main script for training the model
├── config.py                 # Central configuration file for the project
├── build-data-sorted.csv     # The full, raw dataset
├── requirements.txt          # Python dependencies
└── README.md
```
