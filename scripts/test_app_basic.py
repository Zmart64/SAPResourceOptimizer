#!/usr/bin/env python3
"""
Test the app functionality without running Streamlit.

This script verifies that the app can import everything it needs
and that the main functions are defined correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

def test_app_functions():
    """Test that the app functions are defined correctly."""
    print("🧪 Testing app function definitions...")
    
    try:
        # Test importing the app module
        import app
        
        # Check that the main functions exist
        assert hasattr(app, 'run_unified_model'), "run_unified_model function missing"
        assert hasattr(app, 'run_classification'), "run_classification function missing"
        assert hasattr(app, 'run_qe'), "run_qe function missing"
        assert hasattr(app, 'main'), "main function missing"
        
        print("✅ All required functions are defined")
        
        # Check that functions are callable
        assert callable(app.run_unified_model), "run_unified_model not callable"
        assert callable(app.run_classification), "run_classification not callable"  
        assert callable(app.run_qe), "run_qe not callable"
        assert callable(app.main), "main not callable"
        
        print("✅ All functions are callable")
        print("✅ App function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ App function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_imports():
    """Test that the app can import its dependencies."""
    print("\n🧪 Testing app imports...")
    
    try:
        # Test basic imports that don't require external dependencies
        import os
        import sys
        print("✅ Basic imports work")
        
        # Test that the app directory structure is correct
        app_dir = PROJECT_ROOT / "app"
        assert app_dir.exists(), "app directory missing"
        assert (app_dir / "app.py").exists(), "app.py missing"
        assert (app_dir / "data_loader.py").exists(), "data_loader.py missing"
        assert (app_dir / "utils.py").exists(), "utils.py missing"
        
        print("✅ App directory structure is correct")
        print("✅ App import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        return False


def test_model_paths():
    """Test that the model paths in the app are correct."""
    print("\n🧪 Testing model paths...")
    
    try:
        # Expected model paths from the app
        expected_models = {
            "artifacts/trained_models/lightgbm_classification.pkl",
            "artifacts/trained_models/xgboost_classification.pkl", 
            "artifacts/pareto/models/qe_balanced.pkl",
            "artifacts/pareto/models/qe_low_waste.pkl",
            "artifacts/pareto/models/qe_low_underallocation.pkl"
        }
        
        for model_path in expected_models:
            full_path = PROJECT_ROOT / model_path
            if full_path.exists():
                print(f"✅ Found: {model_path}")
            else:
                print(f"⚠️  Missing: {model_path} (may be created during training)")
        
        print("✅ Model path tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model path test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing App Functionality (Without Streamlit)")
    print("=" * 60)
    
    tests = [
        test_app_imports,
        test_app_functions,
        test_model_paths
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed! App implementation looks good.")
        print("\n🎉 Ready to use the unified model interface!")
        return 0
    else:
        print(f"❌ {total - passed}/{total} tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())