#!/usr/bin/env python3
"""
Test the simplified model registration system.

This test validates that the new dynamic model instantiation system works correctly
and reduces the complexity of adding new models.
"""

import pytest
import pandas as pd
import numpy as np
from resource_prediction.config import Config, _import_model_class
from resource_prediction.training.hyperparameter import OptunaOptimizer
from resource_prediction.models.base import BasePredictor


class MockTestModel(BasePredictor):
    """A mock model for testing the simplified registration system."""
    
    def __init__(self, param1: int = 100, param2: float = 0.1, random_state: int = 42, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.random_state = random_state
        self.is_fitted = False
        self.mean_prediction = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> None:
        """Fit the model (dummy implementation)."""
        self.mean_prediction = y.mean()
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (dummy implementation)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(len(X), self.mean_prediction)


class TestSimplifiedModelRegistration:
    """Test the simplified model registration system."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        
        # Create minimal feature set matching Config.BASE_FEATURES
        X = pd.DataFrame({
            'location': np.random.choice(['A', 'B'], 30),
            'component': np.random.choice(['comp1', 'comp2'], 30),
            'makeType': np.random.choice(['debug', 'release'], 30),
            'bp_arch': np.random.choice(['x64', 'x86'], 30),
            'bp_compiler': np.random.choice(['gcc', 'clang'], 30),
            'bp_opt': np.random.choice(['O2', 'O3'], 30),
            'ts_year': [2023] * 30,
            'ts_month': np.random.randint(1, 13, 30),
            'ts_dow': np.random.randint(0, 7, 30),
            'ts_hour': np.random.randint(0, 24, 30),
            'ts_weekofyear': np.random.randint(1, 53, 30),
            'branch_prefix': np.random.choice(['main', 'dev'], 30),
            'jobs': np.random.randint(1, 8, 30),
            'localJobs': np.random.randint(1, 4, 30),
            'target_cnt': np.random.randint(1, 50, 30),
            'target_has_dist': np.random.choice([True, False], 30),
            'branch_id_str': [f'branch_{i}' for i in range(30)],
            'lag_1_grouped': np.random.uniform(1.0, 5.0, 30),
            'lag_max_rss_global_w5': np.random.uniform(2.0, 10.0, 30),
            'rolling_p95_rss_g1_w5': np.random.uniform(3.0, 15.0, 30),
            'build_load': np.random.uniform(0.1, 1.0, 30),
            'target_intensity': np.random.uniform(0.1, 1.0, 30),
            'debug_multiplier': np.random.uniform(1.0, 2.0, 30),
            'heavy_target_flag': np.random.choice([True, False], 30),
            'high_parallelism': np.random.choice([True, False], 30),
        })
        
        y = pd.DataFrame({
            'max_rss_gb': np.random.uniform(1.0, 10.0, 30)
        })
        
        return X, y
    
    def test_dynamic_import_function(self):
        """Test that the dynamic import function works correctly."""
        # Test successful import
        model_class = _import_model_class("resource_prediction.models", "XGBoostRegressor")
        assert model_class.__name__ == "XGBoostRegressor"
        
        # Test import error handling
        with pytest.raises(ImportError):
            _import_model_class("nonexistent.module", "NonexistentClass")
    
    def test_model_families_dynamic_imports(self):
        """Test that all model families load correctly with dynamic imports."""
        config = Config()
        
        for family_name, metadata in config.MODEL_FAMILIES.items():
            model_class = metadata['class']
            assert hasattr(model_class, '__name__'), f"Model class for {family_name} should have a name"
            assert issubclass(model_class, BasePredictor), f"Model {model_class.__name__} should inherit from BasePredictor"
    
    def test_dynamic_model_instantiation_in_hyperparameter_search(self, sample_data):
        """Test that the hyperparameter search can instantiate models dynamically."""
        X, y = sample_data
        
        # Create test config with minimal trials
        class TestConfig(Config):
            N_CALLS_PER_FAMILY = 1
            CV_SPLITS = 2
        
        config = TestConfig()
        
        # Test a few different model families
        test_families = ['lightgbm_regression', 'xgboost_classification', 'qe_regression']
        
        for family_name in test_families:
            optimizer = OptunaOptimizer(config, X, y, model_families=[family_name])
            
            # Test that we can get model metadata and class
            metadata = config.MODEL_FAMILIES[family_name]
            model_class = metadata['class']
            task_type = metadata['type']
            
            assert 'class' in metadata, f"Family {family_name} should have class reference"
            assert 'type' in metadata, f"Family {family_name} should have task type"
            assert task_type in ['regression', 'classification'], f"Invalid task type for {family_name}"
            
            # Test parameter generation
            import optuna
            study = optuna.create_study(direction='minimize')
            trial = study.ask()
            
            params = config.get_search_space(trial, family_name)
            assert 'use_quant_feats' in params, f"Should have use_quant_feats parameter for {family_name}"
            
            # Test dynamic model creation (the key improvement)
            use_quant_feats = params.pop('use_quant_feats')
            model = model_class(**params, random_state=42)
            
            assert isinstance(model, BasePredictor), f"Model {model.__class__.__name__} should be BasePredictor instance"
    
    def test_no_hardcoded_model_creation(self, sample_data):
        """Test that the _objective method works without hardcoded model creation."""
        X, y = sample_data
        
        class TestConfig(Config):
            N_CALLS_PER_FAMILY = 1
            CV_SPLITS = 2
        
        config = TestConfig()
        optimizer = OptunaOptimizer(config, X, y)
        
        # Test that _objective can handle different model families dynamically
        test_families = ['lightgbm_regression', 'rf_classification']
        
        import optuna
        
        for family_name in test_families:
            study = optuna.create_study(direction='minimize')
            trial = study.ask()
            
            # This should work without any hardcoded model-specific logic
            score = optimizer._objective(trial, family_name)
            
            assert isinstance(score, (int, float)), f"Score should be numeric for {family_name}"
            assert score >= 0, f"Score should be non-negative for {family_name}"
    
    def test_mock_model_integration(self):
        """Test that a new model can be integrated with minimal configuration."""
        # Simulate adding a new model with the simplified system
        
        # Step 1: Model class (already created as MockTestModel)
        
        # Step 2: Add to MODEL_FAMILIES (simulate)
        test_family_config = {
            "mock_test_regression": {
                "type": "regression",
                "base_model": "mock_test",
                "class": MockTestModel,
            }
        }
        
        # Step 3: Add hyperparameter config (simulate)
        test_hyperparam_config = {
            "mock_test_regression": {
                "use_quant_feats": {"choices": [True, False], "default": True},
                "param1": {"min": 50, "max": 200, "type": "int", "default": 100},
                "param2": {"min": 0.01, "max": 0.5, "type": "float", "log": True, "default": 0.1},
            }
        }
        
        # Test dynamic instantiation
        model_class = test_family_config["mock_test_regression"]["class"]
        test_params = {"param1": 150, "param2": 0.05}
        model = model_class(**test_params, random_state=42)
        
        assert isinstance(model, MockTestModel)
        assert model.param1 == 150
        assert model.param2 == 0.05
        
        # Test model functionality
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        y_test = pd.Series([1.0, 2.0, 3.0])
        
        model.fit(X_test, y_test)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred == 2.0 for pred in predictions)  # Should be mean of y_test
    
    def test_backward_compatibility(self, sample_data):
        """Test that existing models still work with the new system."""
        X, y = sample_data
        
        config = Config()
        
        # Test that all existing model families are still accessible
        expected_families = {
            'qe_regression', 'xgboost_classification', 'xgboost_regression',
            'lightgbm_classification', 'rf_classification', 'lightgbm_regression',
            'lr_classification', 'sizey_regression'
        }
        
        actual_families = set(config.MODEL_FAMILIES.keys())
        assert expected_families.issubset(actual_families), "Some expected model families are missing"
        
        # Test that hyperparameter configs exist for all families
        for family_name in expected_families:
            assert family_name in config.HYPERPARAMETER_CONFIGS, f"Missing hyperparameter config for {family_name}"
        
        # Test that we can still get default parameters
        for family_name in expected_families:
            defaults = config.get_defaults(family_name)
            assert isinstance(defaults, dict), f"Defaults should be dict for {family_name}"
            assert 'use_quant_feats' in defaults, f"Should have use_quant_feats default for {family_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])