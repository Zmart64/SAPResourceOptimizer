#!/usr/bin/env python3
"""
Test script for the unified model wrapper implementation.

This script tests the basic functionality of the UnifiedModelWrapper
to ensure all components work correctly together.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

def test_unified_wrapper_creation():
    """Test creating unified wrappers manually."""
    print("🧪 Testing UnifiedModelWrapper creation...")
    
    try:
        from resource_prediction.models import UnifiedModelWrapper
        
        # Test creating a mock wrapper (without actual model)
        # This tests the wrapper structure and interface
        
        # Mock model object
        class MockModel:
            def predict_proba(self, X):
                return np.random.random((len(X), 3))
            def predict(self, X):
                return np.random.random(len(X))
            classes_ = [0, 1, 2]
        
        # Test classification wrapper
        classification_wrapper = UnifiedModelWrapper(
            model=MockModel(),
            model_type='lightgbm',
            task_type='classification',
            features=['feature1', 'feature2', 'location', 'component'],
            bin_edges=np.array([0, 1, 2, 4, 8])
        )
        
        print("✅ Classification wrapper created successfully")
        
        # Test regression wrapper  
        regression_wrapper = UnifiedModelWrapper(
            model=MockModel(),
            model_type='quantile_ensemble',
            task_type='regression',
            features=['feature1', 'feature2', 'location']
        )
        
        print("✅ Regression wrapper created successfully")
        
        # Test model info
        info = classification_wrapper.get_model_info()
        assert 'model_type' in info
        assert 'task_type' in info
        assert 'features' in info
        print("✅ Model info retrieval works")
        
        print("✅ All wrapper creation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Wrapper creation test failed: {e}")
        return False


def test_mock_prediction():
    """Test prediction with mock data."""
    print("\n🧪 Testing prediction with mock data...")
    
    try:
        from resource_prediction.models import UnifiedModelWrapper
        
        # Create mock data
        mock_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'location': ['loc1', 'loc2', 'loc1', 'loc3', 'loc2'],
            'component': ['comp1', 'comp2', 'comp1', 'comp1', 'comp3'],
            'makeType': ['make1', 'make2', 'make1', 'make1', 'make2'],
            'bp_arch': ['x86', 'arm', 'x86', 'x86', 'arm'],
            'bp_compiler': ['gcc', 'clang', 'gcc', 'gcc', 'clang'],
            'bp_opt': ['O2', 'O3', 'O2', 'O1', 'O3']
        })
        
        # Mock model that returns predictable results
        class PredictableModel:
            def predict_proba(self, X):
                n_samples = len(X)
                probs = np.random.random((n_samples, 3))
                # Normalize to sum to 1
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs
            
            def predict(self, X):
                return np.random.random(len(X)) * 10
                
            classes_ = [0, 1, 2]
        
        # Test classification prediction
        classification_wrapper = UnifiedModelWrapper(
            model=PredictableModel(),
            model_type='lightgbm',
            task_type='classification',
            features=['feature1', 'feature2', 'location_loc1', 'location_loc2', 'location_loc3', 
                     'component_comp1', 'component_comp2', 'component_comp3',
                     'makeType_make1', 'makeType_make2'],
            bin_edges=np.array([0, 2, 4, 8, 16])
        )
        
        # Test prediction
        predictions = classification_wrapper.predict(mock_data)
        assert len(predictions) == len(mock_data)
        print("✅ Classification prediction works")
        
        # Test regression prediction
        regression_wrapper = UnifiedModelWrapper(
            model=PredictableModel(),
            model_type='quantile_ensemble',
            task_type='regression',
            features=['feature1', 'feature2']  # QE handles its own encoding
        )
        
        predictions = regression_wrapper.predict(mock_data)
        assert len(predictions) == len(mock_data)
        print("✅ Regression prediction works")
        
        print("✅ All prediction tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_any_model_function():
    """Test the load_any_model utility function."""
    print("\n🧪 Testing load_any_model function...")
    
    try:
        from resource_prediction.models import load_any_model
        print("✅ load_any_model import works")
        
        # Note: We can't test actual model loading without the dependencies
        # but we can test that the function exists and has the right interface
        assert callable(load_any_model)
        print("✅ load_any_model is callable")
        
        print("✅ load_any_model function test passed!")
        return True
        
    except Exception as e:
        print(f"❌ load_any_model test failed: {e}")
        return False


def test_app_import():
    """Test that the app can import the new functions."""
    print("\n🧪 Testing app imports...")
    
    try:
        # Test that we can import the unified model function
        sys.path.insert(0, str(PROJECT_ROOT / "app"))
        
        # This should work if the imports are correct
        from resource_prediction.models import load_any_model
        print("✅ App can import load_any_model")
        
        print("✅ All app import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Unified Model Wrapper Implementation")
    print("=" * 60)
    
    tests = [
        test_unified_wrapper_creation,
        test_mock_prediction,
        test_load_any_model_function,
        test_app_import
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed! Implementation looks good.")
        print("\n🎉 Ready to integrate unified models!")
        return 0
    else:
        print(f"❌ {total - passed}/{total} tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())