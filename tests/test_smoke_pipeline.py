import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import pandas as pd
import numpy as np

# Ensure non-interactive backend for matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

# Import the pipeline entrypoint and Config
import main as pipeline_main
from resource_prediction.config import Config


def _make_tiny_raw_csv(path: Path, n_rows: int = 60):
    """Create a tiny synthetic raw CSV with the required schema for preprocessing."""
    rng = np.random.default_rng(42)

    times = pd.date_range("2023-01-01", periods=n_rows, freq="H")
    components = ["compA", "compB", "compC"]
    archs = ["x86_64", "aarch64"]
    compilers = ["gcc", "clang"]
    opts = ["opt", "debug"]
    make_types = ["Ninja", "Make"]
    locations = ["ny", "la", "eu"]

    df = pd.DataFrame(
        {
            "time": times,
            # bytes scale (1GB..3GB) to land in reasonable ranges after division
            "max_rss": rng.integers(low=1 * 1024**3, high=3 * 1024**3, size=n_rows),
            # memreq assumed MB in preprocessing (divided by 1024 -> GB)
            "memreq": rng.choice([512, 1024, 2048, 3072, 4096], size=n_rows),
            "memory_fail_count": rng.choice([0, 0, 0, 1], size=n_rows),
            "buildProfile": [f"{rng.choice(archs)}-{rng.choice(compilers)}-{rng.choice(opts)}" for _ in range(n_rows)],
            "branch": [f"feature_{i%5}_{100+i}" for i in range(n_rows)],
            "targets": ["dist,all,install" if i % 5 == 0 else "lib,unit,test" for i in range(n_rows)],
            "jobs": rng.integers(1, 6, size=n_rows),
            "localJobs": rng.integers(1, 6, size=n_rows),
            "location": [rng.choice(locations) for _ in range(n_rows)],
            "component": [rng.choice(components) for _ in range(n_rows)],
            "makeType": [rng.choice(make_types) for _ in range(n_rows)],
        }
    )

    # Use semicolon separator per preprocessor expectation
    df.to_csv(path, sep=";", index=False)


def _shrink_hyperparameter_configs():
    """Drastically shrink the search space and defaults to make training very fast."""
    # Make CV super small and reduce parallelism
    Config.CV_SPLITS = 2
    Config.N_CALLS_PER_FAMILY = 1
    Config.NUM_PARALLEL_WORKERS = 1
    Config.TEST_SET_FRACTION = 0.2

    # Reduce ranges/defaults for all heavy params
    for fam, params in Config.HYPERPARAMETER_CONFIGS.items():
        for pname, pconf in params.items():
            if not isinstance(pconf, dict):
                continue
            name = pname.lower()
            ptype = pconf.get("type")
            # Consolidate common heavy knobs
            if ptype == "int":
                if "n_estimators" in name or "iterations" in name:
                    pconf["min"] = pconf["max"] = pconf["default"] = 5
                elif "max_depth" in name:
                    pconf["min"] = pconf["max"] = pconf["default"] = 2
                elif "num_leaves" in name:
                    pconf["min"] = pconf["max"] = pconf["default"] = 15
                elif "n_bins" in name:
                    pconf["min"] = pconf["max"] = pconf["default"] = 3
            elif ptype == "float":
                if name.endswith("lr") or name == "learning_rate":
                    pconf["min"] = pconf["max"] = 0.1
                    pconf["default"] = 0.1
                elif "alpha" in name or "quantile" in name:
                    # Keep a single common value to avoid exploring too much
                    pconf["min"] = pconf.get("min", 0.9)
                    pconf["max"] = pconf.get("max", 0.9)
                    pconf["default"] = pconf.get("default", 0.9)
            # Choices left intact; they don't heavily affect runtime


def _configure_paths(tmp_root: Path):
    """Point all Config paths to a temporary directory tree."""
    # Paths
    processed = tmp_root / "processed"
    output = tmp_root / "experiments"
    models = tmp_root / "trained_models"
    optuna_db = output / "optuna_db"

    # Assign config targets
    Config.PROCESSED_DATA_DIR = processed
    Config.OUTPUT_DIR = output
    Config.MODELS_DIR = models
    Config.OPTUNA_DB_DIR = optuna_db
    Config.REGRESSION_RESULTS_CSV_PATH = output / "regression_results.csv"
    Config.CLASSIFICATION_RESULTS_CSV_PATH = output / "classification_results.csv"
    Config.ALLOCATION_PLOT_PATH = output / "memory_allocation_plot.png"
    Config.ALLOCATION_SUMMARY_REPORT_PATH = output / "allocation_summary_report.csv"
    Config.RESULTS_PLOT_PATH = output / "comparison_chart.png"
    Config.SCORE_TIME_PLOT_PATH = output / "score_vs_prediction_time.png"

    # Also redirect processed file paths
    Config.BASELINE_STATS_PATH = processed / "baseline_allocation_stats.pkl"
    Config.X_TRAIN_PATH = processed / "X_train.pkl"
    Config.Y_TRAIN_PATH = processed / "y_train.pkl"
    Config.X_TEST_PATH = processed / "X_test.pkl"
    Config.Y_TEST_PATH = processed / "y_test.pkl"

    # Ensure directories exist
    processed.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    optuna_db.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)


def _run_one(mode: str, include_all_qe_models: bool = False):
    """Run one pipeline mode in an isolated temp dir and validate outputs."""
    assert mode in {"search", "default"}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        _configure_paths(tmp_root)

        # Create tiny raw dataset and point config to it
        raw_path = tmp_root / "build-data-tiny.csv"
        _make_tiny_raw_csv(raw_path, n_rows=60)
        Config.RAW_DATA_PATH = raw_path

        # Shrink search spaces for speed
        _shrink_hyperparameter_configs()

        # Build CLI-like args for main.main
        # Build model family list excluding 'sizey_regression'
        all_families = sorted(Config.MODEL_FAMILIES.keys())
        families_no_sizey = [m for m in all_families if m != "sizey_regression"]

        # Build model family list excluding 'sizey_regression'
        all_families = sorted(Config.MODEL_FAMILIES.keys())
        families_no_sizey = [m for m in all_families if m != "sizey_regression"]

        # Mirror main.py's experimental QE ensembles list for accurate reporting
        experimental_qe_ensembles = [
            'lgb_xgb_ensemble', 'gb_lgb_ensemble', 'xgb_cat_ensemble', 'lgb_cat_ensemble',
            'xgb_xgb_ensemble'
        ]
        all_qe_models = ['gb_xgb_ensemble'] + experimental_qe_ensembles

        # Effective models that main() will run given our args
        if include_all_qe_models:
            effective_models = sorted(set(families_no_sizey).union(all_qe_models))
        else:
            effective_models = sorted(families_no_sizey)

        print("\n--- Model selection summary ---")
        print(f"Mode: {mode}, include_all_qe_models={include_all_qe_models}")
        print(f"Total models to run: {len(effective_models)}")
        print("Models:")
        for m in effective_models:
            print(f"  - {m}")
        print("--- End selection summary ---\n")

        args = SimpleNamespace(
            run_search=(mode == "search"),
            train_default=(mode == "default"),
            evaluate_only=False,
            preprocess_only=False,
            skip_preprocessing=False,
            evaluate_all_archs=False,  # will be overridden by run_optimization_and_evaluation
            task_type=None,
            save_models=False,
            # Explicitly exclude sizey_regression from model families
            model_families=families_no_sizey,
            use_defaults=False,
            run_all_qe_models=include_all_qe_models,
        )

        # Run the pipeline
        print(f"\n=== Running pipeline mode: {mode} (all_qe={include_all_qe_models}) ===")
        pipeline_main.main(args)

        # Validate outputs
        assert Config.REGRESSION_RESULTS_CSV_PATH.exists(), "Missing regression_results.csv"
        assert Config.CLASSIFICATION_RESULTS_CSV_PATH.exists(), "Missing classification_results.csv"

        # Read and ensure non-empty content
        reg_df = pd.read_csv(Config.REGRESSION_RESULTS_CSV_PATH)
        cls_df = pd.read_csv(Config.CLASSIFICATION_RESULTS_CSV_PATH)
        assert not reg_df.empty, "regression_results.csv is empty"
        assert not cls_df.empty, "classification_results.csv is empty"
        assert "model" in reg_df.columns and "score_cv" in reg_df.columns, "Expected columns missing in regression results"
        assert "model" in cls_df.columns and "score_cv" in cls_df.columns, "Expected columns missing in classification results"
        # Classification runs should include tuned confidence_threshold in final params
        assert "confidence_threshold" in cls_df.columns, "classification results must include confidence_threshold"

        # Reporting artifacts should exist because preprocessing creates baseline stats
        assert Config.ALLOCATION_SUMMARY_REPORT_PATH.exists(), "Missing allocation summary report"
        assert Config.ALLOCATION_PLOT_PATH.exists(), "Missing allocation plot"

        # Final comparison plots (only generated when both tasks have results and evaluate_all)
        # They should exist for our default model set as both task types are included.
        assert Config.RESULTS_PLOT_PATH.exists(), "Missing comparison chart"
        assert Config.SCORE_TIME_PLOT_PATH.exists(), "Missing score vs prediction time plot"

        if mode == "search":
            # Optuna DB files should be created
            db_files = list(Config.OPTUNA_DB_DIR.glob("*.db"))
            assert len(db_files) > 0, "No Optuna DB files were created during search"

        print(f"Outputs validated in: {tmp_root}")


def test_hyperparameter_search_smoke():
    """Smoke test: run --run-search on default model set with a tiny dataset and 1 trial."""
    # Run with all experimental QE models included
    _run_one(mode="search", include_all_qe_models=True)


def test_train_default_smoke():
    """Smoke test: run --train-default on default model set with the tiny dataset."""
    # Run with all experimental QE models included
    _run_one(mode="default", include_all_qe_models=True)


if __name__ == "__main__":
    # Allow running as a simple script without pytest
    try:
        test_hyperparameter_search_smoke()
        test_train_default_smoke()
        print("\nAll smoke tests passed.")
    except AssertionError as e:
        print(f"\nSmoke test failed: {e}")
        raise
