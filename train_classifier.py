import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import os
import ast  # To parse string representation of list
import joblib  # For saving the model

# --- Model Imports ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Progress Bar Imports ---
try:
    from tqdm_joblib import tqdm_joblib
    from tqdm.auto import tqdm
except ImportError:
    print("tqdm and tqdm_joblib not found. Progress bars will not be shown for GridSearchCV.")
    print("Install with: pip install tqdm tqdm_joblib")
    tqdm_joblib = None


# --- Suppress Warnings ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
DATA_FILE = "build-data-sorted.csv"
SUMMARY_FILE = os.path.join(
    "", "best_strategy_summary.csv")
TARGET_COLUMN = "max_rss"
N_SPLITS_CV = 3
GRID_SEARCH_N_JOBS = -1
OUTPUT_DIR = "output_plots_best_performer"
TRAIN_FRACTION = 0.8  # NEW: Fraction of data to use for training


# --- Model Configuration for Hyperparameter Search ---
MODEL_CONFIG = {
    'xgboost': {
        'estimator': XGBClassifier,
        'param_grid': {
            'n_estimators': [200, 400], 'learning_rate': [0.03, 0.05],
            'max_depth': [5, 6, 7], 'subsample': [0.7, 0.8], 'colsample_bytree': [0.7, 0.8],
        }, 'requires_one_hot': False, 'fit_params': {}
    },
    'lightgbm': {
        'estimator': LGBMClassifier,
        'param_grid': {
            'n_estimators': [200, 400], 'learning_rate': [0.03, 0.05],
            'max_depth': [5, 7, -1], 'num_leaves': [20, 31, 40], 'subsample': [0.7, 0.8],
        }, 'requires_one_hot': False, 'fit_params': {}
    },
    'catboost': {
        'estimator': CatBoostClassifier,
        'param_grid': {
            'iterations': [200, 400], 'learning_rate': [0.03, 0.05], 'depth': [5, 6, 7],
        }, 'requires_one_hot': False, 'fit_params': {'verbose': 0}
    },
    'random_forest': {
        'estimator': RandomForestClassifier,
        'param_grid': {
            'n_estimators': [150, 300], 'max_depth': [8, 12, 15],
            'min_samples_leaf': [3, 5, 7], 'max_features': ['sqrt', 0.5]
        }, 'requires_one_hot': True, 'fit_params': {}
    },
    'logistic_regression': {
        'estimator': LogisticRegression,
        'param_grid': {
            'C': [0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [1000, 2000]
        }, 'requires_one_hot': True, 'fit_params': {}
    }
}


# --- Helper Functions ---
def split_build_profile(profile_string):
    if not isinstance(profile_string, str):
        return pd.Series(["unknown"]*3)
    parts = profile_string.split('-')
    return pd.Series([parts[0], parts[1] if len(parts) > 1 else "unknown", parts[2] if len(parts) > 2 else "unknown"])

# --- Main Script ---


# 1. Load Best Performer Info
print("--- 1. Loading Best Performer Configuration ---")
try:
    summary_df = pd.read_csv(SUMMARY_FILE)
    best_performer_info = summary_df.iloc[0]
    best_model_name = best_performer_info['model_name']
    bin_edges_gb = ast.literal_eval(best_performer_info['bin_edges'])
    print(f"Best model found: '{best_model_name}'")
    print(f"Using bin edges (GB): {np.round(bin_edges_gb, 2)}")
except Exception as e:
    print(
        f"ERROR: Could not parse '{SUMMARY_FILE}'. It might be empty or in the wrong format. Details: {e}")
    exit()

output_dir_final = os.path.join(OUTPUT_DIR, f"{best_model_name}_final_tuned")
os.makedirs(output_dir_final, exist_ok=True)
print(f"Artifacts will be saved to: {os.path.abspath(output_dir_final)}")


# 2. Load and Prepare Data
print("\n--- 2. Loading and Preparing Data ---")
df = pd.read_csv(DATA_FILE, sep=";")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
df.dropna(subset=[TARGET_COLUMN], inplace=True)
df['max_rss_gb'] = df[TARGET_COLUMN] / (1024**3)

# 3. Feature Engineering
print("--- 3. Starting feature engineering ---")
F = df.copy()
# (Identical feature engineering block as before)
F["ts_year"] = F["time"].dt.year
F["ts_month"] = F["time"].dt.month
F["ts_dow"] = F["time"].dt.dayofweek
F["ts_hour"] = F["time"].dt.hour
F["ts_dayofyear"] = F["time"].dt.dayofyear
F["ts_weekofyear"] = F["time"].dt.isocalendar().week.astype(int)
F[["bp_arch", "bp_compiler", "bp_opt"]
  ] = F["buildProfile"].apply(split_build_profile)
F["branch_id_str"] = F["branch"].str.extract(r"(\d+)$")[0]
F["branch_prefix"] = F["branch"].str.replace(
    r"[\d_]*$", "", regex=True).replace('', 'unknown_prefix')
F["target_cnt"] = F["targets"].astype(str).str.count(",")+1
F["target_has_dist"] = F["targets"].astype(
    str).str.contains("dist").astype("int8")
lag_group_cols = ["component", "bp_arch", "bp_compiler", "bp_opt", "makeType"]
[F.__setitem__(c, F[c].fillna('unknown_in_group_key')
               if c in F.columns else 'unknown_group_val') for c in lag_group_cols]
lag_col_name = f"lag_max_rss_g1_w1"
F[lag_col_name] = F.groupby(lag_group_cols, observed=True)["max_rss"].transform(
    lambda s: s.shift(1).rolling(window=1, min_periods=1).mean())
F["lag_max_rss_global_w5"] = F["max_rss"].shift(
    1).rolling(window=5, min_periods=1).mean()
categorical_features = ["location", "branch_prefix", "bp_arch", "bp_compiler", "bp_opt", "makeType",
                        "component", "ts_year", "ts_month", "ts_dow", "ts_hour", "ts_dayofyear", "ts_weekofyear"]
numerical_features = ["jobs", "localJobs", "target_cnt",
                      "target_has_dist", lag_col_name, "lag_max_rss_global_w5"]
F["branch_id_str"] = pd.to_numeric(
    F["branch_id_str"], errors='coerce').fillna(-1)
numerical_features.append("branch_id_str")
features_to_use = sorted(list(set(F.columns) & set(
    categorical_features+numerical_features)))
X_intermediate = F[features_to_use].copy()

# 4. Target Variable Creation & Final Data Prep
print("\n--- 4. Creating Target and Finalizing Data ---")
num_target_classes = len(bin_edges_gb) - 1
target_class_labels_idx = list(range(num_target_classes))
y = pd.cut(F['max_rss_gb'], bins=bin_edges_gb, labels=target_class_labels_idx,
           right=False, include_lowest=True, duplicates='drop')
valid_indices = y.dropna().index
X_intermediate = X_intermediate.loc[valid_indices]
F = F.loc[valid_indices]
y = y.loc[valid_indices].astype(int)
config = MODEL_CONFIG[best_model_name]
if config['requires_one_hot']:
    X = pd.get_dummies(X_intermediate, columns=X_intermediate.select_dtypes(
        include='object').columns.tolist(), drop_first=True)
else:
    X = X_intermediate.copy()
    [X.__setitem__(c, X[c].astype('category'))
     for c in X.select_dtypes(include=['object', 'category']).columns]
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())
F_eval = F.copy()
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
F_eval = F_eval.reset_index(drop=True)
print(f"Total processed samples: {len(X)}")

# 5. Split Data into Training (80%) and Holdout/Simulation (20%)
print(
    f"\n--- 5. Splitting Data into Training ({TRAIN_FRACTION*100:.0f}%) and Holdout/Simulation ({(1-TRAIN_FRACTION)*100:.0f}%) ---")
split_index = int(len(X) * TRAIN_FRACTION)

# The first 80% of the data is for training and tuning
X_train_full = X.iloc[:split_index].copy()
y_train_full = y.iloc[:split_index].copy()
F_train_full = F_eval.iloc[:split_index].copy()

# The last 20% is the holdout set for simulation
X_holdout = X.iloc[split_index:].copy()
y_holdout = y.iloc[split_index:].copy()
F_holdout = F_eval.iloc[split_index:].copy()

print(f"Training set size: {len(X_train_full)}")
print(f"Holdout set size: {len(X_holdout)}")

# Save the holdout data for the simulation part. This dataframe contains the
# final features for prediction and the true values for evaluation.
sim_data = X_holdout.copy()
sim_data['time'] = F_holdout['time'].values
sim_data['max_rss_gb_true'] = F_holdout['max_rss_gb'].values
sim_data['max_rss_class_true'] = y_holdout.values

simulation_data_path = os.path.join(
    OUTPUT_DIR, "simulation_data.csv")
sim_data.to_csv(simulation_data_path, index=False, sep=';')
print(
    f"Holdout data for simulation saved to: {os.path.abspath(simulation_data_path)}")


# 6. Hyperparameter Search on Training Data
print("\n--- 6. Performing Hyperparameter Search on Training Data ---")
tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
base_model = config['estimator'](random_state=42)
param_grid = config['param_grid']
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid,
                           scoring='accuracy', cv=tscv, verbose=0, n_jobs=GRID_SEARCH_N_JOBS)
if tqdm_joblib and GRID_SEARCH_N_JOBS != 1:
    with tqdm_joblib(tqdm(total=np.prod([len(v) for v in param_grid.values()])*N_SPLITS_CV, desc="GridSearch Progress")) as pbar:
        grid_search.fit(X_train_full, y_train_full)
else:
    grid_search.fit(X_train_full, y_train_full)
best_params_from_grid = grid_search.best_params_
print("\n--- GridSearchCV Results ---")
print(f"Best parameters found: {best_params_from_grid}")

# 7. Evaluate Final Model with Cross-Validation on Training Data
print("\n--- 7. Evaluating Final Tuned Model with Cross-Validation on Training Data ---")
final_model_for_eval = config['estimator'](
    random_state=42, **best_params_from_grid)

# --- Initialize accumulators for detailed reporting ---
all_y_true, all_y_pred = [], []
overall_total_jobs_evaluated = 0
overall_total_true_used_gb = 0
overall_total_allocated_gb = 0
overall_total_over_allocated_gb = 0
overall_total_under_allocated_gb = 0
overall_jobs_perfectly_allocated = 0
overall_jobs_under_allocated = 0
overall_jobs_well_allocated_over = 0
overall_jobs_severe_over_2x = 0
overall_jobs_extreme_over_3x = 0
overall_jobs_massive_over_4x = 0

for fold, (train_index, test_index) in enumerate(tscv.split(X_train_full, y_train_full)):
    X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
    y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]
    fit_params = config['fit_params'].copy()
    if best_model_name == 'catboost':
        fit_params['cat_features'] = X_train.select_dtypes(
            'category').columns.tolist()
    final_model_for_eval.fit(X_train, y_train, **fit_params)
    y_pred = final_model_for_eval.predict(X_test)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    # --- Detailed Allocation Metrics Calculation ---
    allocated_mem_pred = np.array(
        [bin_edges_gb[min(c + 1, len(bin_edges_gb) - 1)] for c in y_pred])
    true_mem = F_train_full.iloc[test_index]['max_rss_gb'].values
    overall_total_jobs_evaluated += len(y_test)
    overall_total_true_used_gb += np.sum(true_mem)
    overall_total_allocated_gb += np.sum(allocated_mem_pred)
    overall_total_over_allocated_gb += np.sum(
        np.maximum(0, allocated_mem_pred - true_mem))

    for i in range(len(y_test)):
        true_val, alloc_val = true_mem[i], allocated_mem_pred[i]
        if alloc_val < true_val:
            overall_jobs_under_allocated += 1
            overall_total_under_allocated_gb += (true_val - alloc_val)
        elif np.isclose(alloc_val, true_val):
            overall_jobs_perfectly_allocated += 1
        else:
            ratio = alloc_val / true_val if true_val > 1e-9 else float('inf')
            if ratio < 2:
                overall_jobs_well_allocated_over += 1
            elif ratio < 3:
                overall_jobs_severe_over_2x += 1
            elif ratio < 4:
                overall_jobs_extreme_over_3x += 1
            else:
                overall_jobs_massive_over_4x += 1

# 8. Train Final Model on ALL Training Data and Save
print("\n--- 8. Training Final Model on All Training Data (80%) for Saving ---")
final_model_to_save = config['estimator'](
    random_state=42, **best_params_from_grid)
fit_params = config['fit_params'].copy()
if best_model_name == 'catboost':
    fit_params['cat_features'] = X_train_full.select_dtypes(
        'category').columns.tolist()
final_model_to_save.fit(X_train_full, y_train_full, **fit_params)

# --- NEW: Create a rich payload to save ---
print("--- Creating model payload with metadata ---")
model_payload = {
    'model': final_model_to_save,
    'model_name': best_model_name,
    'bin_edges_gb': bin_edges_gb,
    'feature_columns': X_train_full.columns.tolist(),  # The exact feature set
    'requires_one_hot': config['requires_one_hot'],
    # Get original categorical columns before one-hot encoding for the simulator
    'original_categorical_features': X_intermediate.select_dtypes(include='object').columns.tolist()
}

model_save_path = os.path.join(output_dir_final, "final_model.pkl")
# --- MODIFIED: Save the entire payload dictionary ---
joblib.dump(model_payload, model_save_path)
print(f"Final model payload saved to: {model_save_path}")


# 9. Aggregate Results and Final Reports
print("\n--- 9. Aggregate Results and Final Reports from CV on Training Data ---")
report_filename = os.path.join(output_dir_final, "final_model_report.txt")
cm_filename = os.path.join(output_dir_final, "final_confusion_matrix.png")
fi_filename = os.path.join(output_dir_final, "final_feature_importance.png")

# --- Calculate final percentages for reporting ---
if overall_total_jobs_evaluated > 0:
    perc_perfect = (overall_jobs_perfectly_allocated /
                    overall_total_jobs_evaluated) * 100
    perc_under = (overall_jobs_under_allocated /
                  overall_total_jobs_evaluated) * 100
    perc_well_over = (overall_jobs_well_allocated_over /
                      overall_total_jobs_evaluated) * 100
    perc_severe_2x = (overall_jobs_severe_over_2x /
                      overall_total_jobs_evaluated) * 100
    perc_extreme_3x = (overall_jobs_extreme_over_3x /
                       overall_total_jobs_evaluated) * 100
    perc_massive_4x = (overall_jobs_massive_over_4x /
                       overall_total_jobs_evaluated) * 100
else:
    perc_perfect = perc_under = perc_well_over = perc_severe_2x = perc_extreme_3x = perc_massive_4x = 0
perc_over_alloc_total = (overall_total_over_allocated_gb /
                         overall_total_true_used_gb) * 100 if overall_total_true_used_gb > 0 else 0

# --- Print formatted output to console ---
print("\n--- Overall Memory Allocation Summary (from CV on Training Set) ---")
print(f"Total Jobs Evaluated in CV: {overall_total_jobs_evaluated}")
print(f"Total True Memory Used (GB): {overall_total_true_used_gb:.2f}")
print(f"Total Memory Allocated (GB): {overall_total_allocated_gb:.2f}")
print(
    f"Total OVER-Allocated Memory (GB): {overall_total_over_allocated_gb:.2f}")
print(
    f"Total UNDER-Allocated Amount (GB, where alloc < rss): {overall_total_under_allocated_gb:.2f}")
print(
    f"Percentage Total Over-allocation relative to True Usage: {perc_over_alloc_total:.2f}%")
print(
    f"\n--- Detailed Allocation Categories ({overall_total_jobs_evaluated} jobs) ---")
print(
    f"Perfectly Allocated (alloc approx= true):      {overall_jobs_perfectly_allocated:>6} jobs ({perc_perfect:6.2f}%)")
print(
    f"Under-allocated (alloc < true):              {overall_jobs_under_allocated:>6} jobs ({perc_under:6.2f}%)")
print(
    f"Well-allocated Over (true <= alloc < 2*true):  {overall_jobs_well_allocated_over:>6} jobs ({perc_well_over:6.2f}%)")
print(
    f"Severely Over-allocated (2*true <= alloc < 3*true): {overall_jobs_severe_over_2x:>6} jobs ({perc_severe_2x:6.2f}%)")
print(
    f"Extremely Over-allocated (3*true <= alloc < 4*true):{overall_jobs_extreme_over_3x:>6} jobs ({perc_extreme_3x:6.2f}%)")
print(
    f"Massively Over-allocated (alloc >= 4*true):    {overall_jobs_massive_over_4x:>6} jobs ({perc_massive_4x:6.2f}%)")


# --- Write formatted output to file ---
with open(report_filename, "w") as f:
    f.write(f"Final Tuned Model Report for: '{best_model_name}'\n")
    f.write(
        f"NOTE: This report is based on TimeSeries-Cross-Validation on the first {TRAIN_FRACTION*100:.0f}% of the data.\n")
    f.write(
        f"The final {100-TRAIN_FRACTION*100:.0f}% of the data was held out for simulation.\n")
    f.write(f"Bin Edges (GB): {np.round(bin_edges_gb, 2)}\n")
    f.write("\n--- Best Hyperparameters from GridSearchCV ---\n")
    f.write(str(best_params_from_grid) + "\n")

    report_str = classification_report(all_y_true, all_y_pred, labels=target_class_labels_idx, target_names=[
                                       f"C{l}" for l in target_class_labels_idx], zero_division=0)
    f.write(
        f"\n--- Classification Report (from CV on Training Set) ---\n{report_str}\n")

    f.write("\n--- Overall Memory Allocation Summary (from CV on Training Set) ---\n")
    f.write(f"Total Jobs Evaluated in CV: {overall_total_jobs_evaluated}\n")
    f.write(f"Total True Memory Used (GB): {overall_total_true_used_gb:.2f}\n")
    f.write(f"Total Memory Allocated (GB): {overall_total_allocated_gb:.2f}\n")
    f.write(
        f"Total OVER-Allocated Memory (GB): {overall_total_over_allocated_gb:.2f}\n")
    f.write(
        f"Total UNDER-Allocated Amount (GB, where alloc < rss): {overall_total_under_allocated_gb:.2f}\n")
    f.write(
        f"Percentage Total Over-allocation relative to True Usage: {perc_over_alloc_total:.2f}%\n")

    f.write(
        f"\n--- Detailed Allocation Categories ({overall_total_jobs_evaluated} jobs) ---\n")
    f.write(
        f"Perfectly Allocated (alloc approx= true):      {overall_jobs_perfectly_allocated:>6} jobs ({perc_perfect:6.2f}%)\n")
    f.write(
        f"Under-allocated (alloc < true):              {overall_jobs_under_allocated:>6} jobs ({perc_under:6.2f}%)\n")
    f.write(
        f"Well-allocated Over (true <= alloc < 2*true):  {overall_jobs_well_allocated_over:>6} jobs ({perc_well_over:6.2f}%)\n")
    f.write(
        f"Severely Over-allocated (2*true <= alloc < 3*true): {overall_jobs_severe_over_2x:>6} jobs ({perc_severe_2x:6.2f}%)\n")
    f.write(
        f"Extremely Over-allocated (3*true <= alloc < 4*true):{overall_jobs_extreme_over_3x:>6} jobs ({perc_extreme_3x:6.2f}%)\n")
    f.write(
        f"Massively Over-allocated (alloc >= 4*true):    {overall_jobs_massive_over_4x:>6} jobs ({perc_massive_4x:6.2f}%)\n")


# Standard plots (CM and Feature Importance)
cm = confusion_matrix(all_y_true, all_y_pred, labels=target_class_labels_idx)
plt.figure(figsize=(max(8, num_target_classes*1.5),
           max(6, num_target_classes*1.2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            f"C{l}" for l in target_class_labels_idx], yticklabels=[f"C{l}" for l in target_class_labels_idx])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'CM on Training CV Folds - {best_model_name}')
plt.tight_layout()
plt.savefig(cm_filename)
plt.close()
if hasattr(final_model_to_save, 'feature_importances_'):
    fi_df = pd.DataFrame({'feature': X_train_full.columns, 'importance': final_model_to_save.feature_importances_}).sort_values(
        'importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=fi_df.head(20))
    plt.title(
        f'Top 20 Feature Importance - {best_model_name} (Trained on 80% Data)')
    plt.tight_layout()
    plt.savefig(fi_filename)
    plt.close()

print(
    f"\nAll artifacts (model, plots, report) saved to '{os.path.abspath(output_dir_final)}'")
print("Script finished.")
