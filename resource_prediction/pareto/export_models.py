"""
Export Pareto key-point models to disk.
"""
import copy
import joblib
import pandas as pd

from resource_prediction.config import Config
from resource_prediction.models.implementations.quantile_ensemble import QuantileEnsemblePredictor
from resource_prediction.models import DeployableModel
from resource_prediction.preprocessing import ModelPreprocessor


def save_models(key_points: dict, config: Config) -> dict:
    """Export three Pareto configurations as serialized DeployableModel files."""
    # Ensure output directory exists
    models_subfolder = config.PROJECT_ROOT / "artifacts" / "pareto" / "models"
    models_subfolder.mkdir(parents=True, exist_ok=True)

    # Load base trained model
    base_path = config.MODELS_DIR / "qe_regression.pkl"
    try:
        base_deployable = joblib.load(base_path)
        if isinstance(base_deployable, DeployableModel):
            base_model = base_deployable.model
            preprocessor = base_deployable.preprocessor
        else:
            raise ValueError
    except Exception:
        data = joblib.load(base_path)
        base_model = data['model']
        X_train = pd.read_pickle(config.X_TRAIN_PATH)
        preprocessor = ModelPreprocessor(
            categorical_features=config.CATEGORICAL_FEATURES,
            numerical_features=config.NUMERICAL_FEATURES,
            target_column=config.TARGET_COLUMN,
            drop_columns=config.COLUMNS_TO_DROP
        )
        preprocessor.fit(X_train)

    # Extract common parameters
    gb_params = base_model.gb.get_params()

    saved = {}
    specs = [
        ('low_waste', 'green'),
        ('low_underallocation', 'blue'),
        ('balanced', 'red')
    ]
    for name, color in specs:
        point = key_points[name]
        # build new predictor
        predictor = QuantileEnsemblePredictor(
            alpha=point['alpha'],
            safety=point['safety'],
            gb_params={k: v for k, v in gb_params.items() if k not in ['alpha','random_state']},
            xgb_params={
                'n_estimators': base_model.xgb.get_params().get('n_estimators', 100),
                'max_depth': base_model.xgb.get_params().get('max_depth', 3),
                'learning_rate': base_model.xgb.get_params().get('learning_rate', 0.1),
                'n_jobs': 1
            },
            random_state=42
        )
        # copy fitted trees and cols
        predictor.columns = base_model.columns
        predictor.gb = copy.deepcopy(base_model.gb)
        predictor.xgb = copy.deepcopy(base_model.xgb)
        predictor.alpha = point['alpha']
        predictor.safety = point['safety']

        # wrap
        deploy = DeployableModel(
            model=predictor,
            model_type='quantile_ensemble',
            task_type='regression',
            preprocessor=preprocessor,
            bin_edges=None,
            metadata={
                'alpha': point['alpha'],
                'safety': point['safety'],
                'waste_pct': point['total_over_pct'],
                'underallocation_pct': point['under_pct'],
                'business_score': point['business_score'],
                'pareto_configuration': True,
                'training_timestamp': str(pd.Timestamp.now())
            }
        )
        out_path = models_subfolder / f"qe_{name}.pkl"
        deploy.save(out_path)
        saved[name] = out_path
    # summary file
    summary = models_subfolder / 'models_summary.txt'
    with open(summary, 'w') as f:
        f.write('Pareto Frontier Model Configurations\n')
        f.write('='*50 + '\n')
        for name, path in saved.items():
            p = key_points[name]
            f.write(f"{name}: file={path.name}, alpha={p['alpha']:.3f}, safety={p['safety']:.3f}\n")
    return saved
