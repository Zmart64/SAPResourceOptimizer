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


def main(args):
    """
    Main function to orchestrate the ML pipeline.

    Handles command-line arguments to run preprocessing, hyperparameter
    optimization, and final model evaluation.
    """
    config = Config()

    if args.preprocess_only or not args.skip_preprocessing:
        preprocessor = DataPreprocessor(config)
        preprocessor.process()
        if args.preprocess_only:
            print("Preprocessing complete. Exiting as requested.")
            return

    if args.run_search:
        trainer = Trainer(
            config,
            evaluate_all_archs=args.evaluate_all_archs,
            task_type_filter=args.task_type
        )
        trainer.run_optimization_and_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main pipeline for the resource prediction project. Handles preprocessing, hyperparameter optimization, and final evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--run-search",
        action="store_true",
        help="Run the full hyperparameter search and final evaluation."
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the data preprocessing step and use existing processed data files."
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run the data preprocessing step and then exit."
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

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    main(args)
