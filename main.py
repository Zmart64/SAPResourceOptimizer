import argparse
import warnings
from resource_prediction.config import Config
from resource_prediction.data_processing.preprocessor import DataPreprocessor
from resource_prediction.training.trainer import Trainer

# Suppress common warnings for a cleaner command-line output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main(args):
    """
    Main function to orchestrate the ML pipeline.

    The workflow is determined by the command-line arguments:
    - If --skip-preprocessing is specified, the preprocessing step is skipped.
    - If --run-optimization is specified, the optimization process is run.
      By default, this is followed by the final evaluation on the test set.
    - The --optimize-only flag can be used with --run-optimization to prevent
      the automatic evaluation step from running.
    - If --evaluate-on-test is specified, only the final evaluation is run,
      using the results from a previous optimization run.
    """
    # Check if any action was requested by the user
    if not args.run_optimization and not args.evaluate_on_test:
        print("Error: No task specified. You must select at least one task to run.")
        parser.print_help()
        return

    config = Config()

    if not args.skip_preprocessing:
        preprocessor = DataPreprocessor(config)
        preprocessor.process()
    else:
        print("Skipping preprocessing as requested. Using existing processed data.")

    trainer = Trainer(config)

    if args.run_optimization:
        trainer.run_bayesian_optimization()

    # Runs either if:
    # - user requests it (--evaluate-on-test)
    # - or if user does not specify --optimize-only
    should_evaluate = args.evaluate_on_test or (
        args.run_optimization and not args.optimize_only)

    if should_evaluate:
        trainer.evaluate_best_model_on_test_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main pipeline for the memory prediction project.\n"
                    "Handles preprocessing, hyperparameter optimization, and final evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- General Options ---
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="If set, skip the preprocessing and data splitting step."
    )

    # --- Task Selection ---
    parser.add_argument(
        "--run-optimization",
        action="store_true",
        help="STAGE 1: Run Bayesian optimization on the training set.\n"
             "By default, this will be followed by a final evaluation (see --optimize-only)."
    )

    parser.add_argument(
        "--evaluate-on-test",
        action="store_true",
        help="STAGE 2: Run a final evaluation on the holdout test set.\n"
             "This uses the 'optimization_summary.csv' from a previous run."
    )

    # --- Modifiers ---
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Use with --run-optimization to ONLY run the optimization step\n"
             "and stop before the final evaluation."
    )

    args = parser.parse_args()
    main(args)
