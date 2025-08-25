# main.py

"""Command-line interface for the resource prediction pipeline."""

import argparse
import os
import sys
import warnings
from resource_prediction.config import Config
from resource_prediction.data_processing.preprocessor import DataPreprocessor
from resource_prediction.training.trainer import Trainer

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[var] = "1"

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Suppress Optuna ExperimentalWarning for multivariate/constant_liar sampler options
try:
    from optuna.exceptions import ExperimentalWarning as _OptunaExperimentalWarning
    warnings.filterwarnings('ignore', category=_OptunaExperimentalWarning)
except Exception:
    pass


def main(args):
    """
    Main function to orchestrate the ML pipeline.

    Handles command-line arguments to run preprocessing, hyperparameter
    optimization, and final model evaluation.
    """
    config = Config()

    if not args.skip_preprocessing and (args.run_search or args.train_default or args.evaluate_only):
        preprocessor = DataPreprocessor(config)
        preprocessor.process()

    if args.preprocess_only:
        preprocessor = DataPreprocessor(config)
        preprocessor.process()
        print("Preprocessing complete. Exiting as requested.")
        return

    # Set use_defaults=True for --train-default option
    use_defaults = getattr(args, 'use_defaults', False)
    if args.train_default:
        use_defaults = True

    # Handle --run-all-qe-models flag
    model_families = args.model_families

    # Define experimental QE ensemble models (exclude the standard lgb_xgb_ensemble)
    # We now treat lgb_xgb_ensemble as the standard/default QE architecture.
    experimental_qe_ensembles = [
        'gb_xgb_ensemble', 'gb_lgb_ensemble', 'xgb_cat_ensemble', 'lgb_cat_ensemble',
        'xgb_xgb_ensemble'
    ]
    all_qe_models = ['lgb_xgb_ensemble'] + experimental_qe_ensembles

    if args.run_all_qe_models:
        # When flag is set:
        # - If user specified model families: union of those + ALL QE models
        # - Otherwise: run ALL available model families plus ALL QE models
        if model_families:
            model_families = list(sorted(set(model_families).union(all_qe_models)))
        else:
            model_families = list(sorted(set(Config.MODEL_FAMILIES.keys()).union(all_qe_models)))
    else:
        # Default behavior without the flag:
        # If user did not specify families: run all except experimental QE ensembles (keep lgb_xgb_ensemble)
        if model_families is None:
            all_models = list(Config.MODEL_FAMILIES.keys())
            model_families = [m for m in all_models if m not in experimental_qe_ensembles]

    print("running model families:", model_families)

    trainer = Trainer(
        config,
        evaluate_all_archs=args.evaluate_all_archs,
        task_type_filter=args.task_type,
        save_models=args.save_models,
        model_families=model_families,
        use_defaults=use_defaults
    )

    if args.run_search or args.train_default:
        trainer.run_optimization_and_evaluation()
    elif args.evaluate_only:
        trainer.run_evaluation_from_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main pipeline for the resource prediction project. Handles preprocessing, hyperparameter optimization, and final evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Group for mutually exclusive actions. One of these is required.
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--run-search",
        action="store_true",
        help="Run the full hyperparameter search and final evaluation."
    )
    action_group.add_argument(
        "--train-default",
        dest="train_default",
        action="store_true",
        help="Train models with default parameters and evaluate them. Similar to --run-search --use-defaults but simpler."
    )
    action_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate models using best parameters from existing results CSV files, skipping the search."
    )
    action_group.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run the data preprocessing step and then exit."
    )

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the data preprocessing step and use existing processed data files."
    )
    parser.add_argument(
        "--evaluate-all-archs",
        action="store_true",
        help="During final evaluation, evaluate the best performer from ALL model architectures\ninstead of just the single best champion for each task type. This will also generate a comparison plot."
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=['regression', 'classification'],
        default=None,
        help="Optionally filter the pipeline to run ONLY for a specific task type (e.g., 'regression').\nBy default, both are run."
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="If set, saves the final evaluated champion model(s) as .pkl files in the `resource_prediction/models` directory."
    )
    parser.add_argument(
        "--model-families",
        type=str,
        nargs="*",
        help="Specify which model families to run (e.g., 'xgboost_regression rf_classification').\nBy default, all model families are run. Available options: " + 
             ", ".join(Config.MODEL_FAMILIES.keys())
    )
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="[Only with --run-search] Train models with default hyperparameters instead of running hyperparameter search.\nThis provides a quick way to get baseline results without optimization. Use --train-default for a simpler alternative."
    )
    parser.add_argument(
        "--run-all-qe-models",
        action="store_true",
        help="Run all experimental quantile ensemble models in addition to the standard gb_xgb_ensemble model.\nBy default, only gb_xgb_ensemble is run. Can be combined with --model-families."
    )

    args = parser.parse_args()

    main(args)
