"""
Export Pareto key-point models to disk.

This version retrains the LGB+XGB QE base learners for each key-point alpha
before saving, so exported models differ by both alpha and safety.
"""
from __future__ import annotations

import joblib
import pandas as pd

from resource_prediction.config import Config
from resource_prediction.models.implementations.quantile_ensemble_variants import LGBXGBQuantileEnsemble
from resource_prediction.models import DeployableModel


def save_models(key_points: dict, config: Config) -> dict:
    """Export three Pareto configurations as serialized DeployableModel files.

    For each key-point, retrain a fresh LGB+XGB QE with the specified alpha and
    reuse the safety value during inference.
    """
    # Output directory
    models_subfolder = config.PROJECT_ROOT / "artifacts" / "pareto" / "models"
    models_subfolder.mkdir(parents=True, exist_ok=True)

    # Load a baseline deployable (for preprocessor + to extract hyperparams)
    base_path = config.MODELS_DIR / "lgb_xgb_ensemble.pkl"
    base_deployable: DeployableModel = joblib.load(base_path)
    if not isinstance(base_deployable, DeployableModel):
        raise ValueError(f"Unexpected model artifact at {base_path}")

    base_model = base_deployable.model  # LGBXGBQuantileEnsemble
    preprocessor = base_deployable.preprocessor

    # Extract tuned hyperparameters from the fitted base learners
    lgb_params = base_model.lgb.get_params()
    xgb_params = base_model.xgb.get_params()

    # Load training data (raw engineered features)
    X_train = pd.read_pickle(config.X_TRAIN_PATH)
    y_train = pd.read_pickle(config.Y_TRAIN_PATH)[config.TARGET_COLUMN_PROCESSED]

    # Use the same raw feature set as during training of the champion model
    # If champion used a subset, the internal encoder will align columns accordingly
    # Default to BASE + QUANT features (preprocessor handles at inference)
    # but passing raw X to the QE wrapper ensures consistent one-hot flow.
    # If you need to force the exact subset, adjust here.

    saved = {}
    for name in ["low_waste", "low_underallocation", "balanced"]:
        point = key_points[name]
        alpha = float(point["alpha"]) if "alpha" in point else 0.95
        safety = float(point["safety"]) if "safety" in point else 1.0

        # Build a fresh predictor with new alpha and tuned hyperparams (excluding alpha fields)
        predictor = LGBXGBQuantileEnsemble(
            alpha=alpha,
            safety=safety,
            lgb_params={k: v for k, v in lgb_params.items() if k not in ["alpha", "random_state"]},
            xgb_params={k: v for k, v in xgb_params.items() if k not in ["quantile_alpha", "random_state"]},
            random_state=config.RANDOM_STATE,
        )

        # Retrain the base learners at the requested alpha
        predictor.fit(X_train, y_train)

        # Wrap with the existing preprocessor to ensure consistent inference
        deploy = DeployableModel(
            model=predictor,
            model_type="lgb_xgb_ensemble",
            task_type="regression",
            preprocessor=preprocessor,
            bin_edges=None,
            metadata={
                "alpha": alpha,
                "safety": safety,
                "waste_pct": float(point.get("total_over_pct", float("nan"))),
                "underallocation_pct": float(point.get("under_pct", float("nan"))),
                "business_score": float(point.get("business_score", float("nan"))),
                "pareto_configuration": True,
                "training_timestamp": str(pd.Timestamp.now()),
            },
        )

        out_path = models_subfolder / f"qe_{name}.pkl"
        deploy.save(out_path)
        saved[name] = out_path

    # Save a simple summary text file
    summary = models_subfolder / "models_summary.txt"
    with open(summary, "w") as f:
        f.write("Pareto Frontier Model Configurations\n")
        f.write("=" * 50 + "\n")
        for name, path in saved.items():
            p = key_points[name]
            f.write(
                f"{name}: file={path.name}, alpha={float(p['alpha']):.3f}, safety={float(p['safety']):.3f}\n"
            )

    return saved
