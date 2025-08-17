#!/usr/bin/env python3
"""
Script to convert existing legacy model files to unified wrapper format.

This script helps migrate all existing models to use the new UnifiedModelWrapper
interface for consistent preprocessing and prediction across all model types.
"""

import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from resource_prediction.models import convert_legacy_models_to_unified


def main():
    """Convert all legacy models to unified wrapper format."""
    
    # Paths
    legacy_models_dir = PROJECT_ROOT / "artifacts" / "trained_models"
    pareto_models_dir = PROJECT_ROOT / "artifacts" / "pareto" / "models"
    unified_models_dir = PROJECT_ROOT / "artifacts" / "unified_models"
    
    print("🔄 Converting Legacy Models to Unified Wrapper Format")
    print("=" * 60)
    
    # Convert main trained models
    if legacy_models_dir.exists():
        print(f"\n📂 Converting models from: {legacy_models_dir}")
        convert_legacy_models_to_unified(legacy_models_dir, unified_models_dir)
    else:
        print(f"❌ Legacy models directory not found: {legacy_models_dir}")
    
    # Convert Pareto models
    if pareto_models_dir.exists():
        print(f"\n📂 Converting Pareto models from: {pareto_models_dir}")
        convert_legacy_models_to_unified(pareto_models_dir, unified_models_dir / "pareto")
    else:
        print(f"❌ Pareto models directory not found: {pareto_models_dir}")
    
    print(f"\n✅ Conversion complete! Unified models saved to: {unified_models_dir}")
    print("\nYou can now use the unified models with the load_any_model() function.")


if __name__ == "__main__":
    main()