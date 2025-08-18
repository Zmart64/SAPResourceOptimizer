#!/usr/bin/env python3
"""
Create a focused Pareto frontier plot with three highlighted key points,
and export the corresponding model pkl files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
import copy
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from resource_prediction.config import Config
from resource_prediction.models.implementations.quantile_ensemble import QuantileEnsemblePredictor
from resource_prediction.models import DeployableModel
from resource_prediction.preprocessing import ModelPreprocessor


def create_focused_pareto_plot():
    """Create a single plot showing only the Pareto frontier with key points highlighted."""
    config = Config()
    
    # Load Pareto frontier points
    pareto_df = pd.read_csv(config.PROJECT_ROOT / "artifacts" / "pareto" / "results" / "pareto_frontier_points.csv")
    
    # Identify the three key points
    min_waste_idx = pareto_df['total_over_pct'].idxmin()
    min_under_idx = pareto_df['under_pct'].idxmin() 
    min_business_idx = pareto_df['business_score'].idxmin()
    
    min_waste = pareto_df.loc[min_waste_idx]
    min_under = pareto_df.loc[min_under_idx]
    min_business = pareto_df.loc[min_business_idx]
    
    print("Three key points identified:")
    print(f"1. Low waste: α={min_waste['alpha']:.2f}, safety={min_waste['safety']:.2f}")
    print(f"   → {min_waste['total_over_pct']:.1f}% waste, {min_waste['under_pct']:.1f}% underallocation")
    print(f"2. Low underallocation: α={min_under['alpha']:.2f}, safety={min_under['safety']:.2f}")
    print(f"   → {min_under['total_over_pct']:.1f}% waste, {min_under['under_pct']:.1f}% underallocation")
    print(f"3. Balanced: α={min_business['alpha']:.2f}, safety={min_business['safety']:.2f}")
    print(f"   → {min_business['total_over_pct']:.1f}% waste, {min_business['under_pct']:.1f}% underallocation")
    
    # Create the focused plot
    plt.figure(figsize=(10, 8))
    
    # Sort Pareto points by waste for connecting line
    pareto_sorted = pareto_df.sort_values('total_over_pct')
    
    # Plot all Pareto frontier points
    plt.scatter(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'], 
               c='lightgray', s=60, alpha=0.7, label='Pareto Frontier Points')
    
    # Plot the Pareto frontier line
    plt.plot(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'], 
            'gray', linewidth=2, alpha=0.8, label='Pareto Frontier')
    
    # Highlight the three key points with different colors and larger sizes
    plt.scatter(min_waste['total_over_pct'], min_waste['under_pct'], 
               c='green', s=200, marker='o', edgecolors='black', linewidth=2,
               label=f'Low Waste (α={min_waste["alpha"]:.2f}, s={min_waste["safety"]:.2f})')
    
    plt.scatter(min_under['total_over_pct'], min_under['under_pct'], 
               c='blue', s=200, marker='s', edgecolors='black', linewidth=2,
               label=f'Low Underallocation (α={min_under["alpha"]:.2f}, s={min_under["safety"]:.2f})')
    
    plt.scatter(min_business['total_over_pct'], min_business['under_pct'], 
               c='red', s=200, marker='^', edgecolors='black', linewidth=2,
               label=f'Balanced (α={min_business["alpha"]:.2f}, s={min_business["safety"]:.2f})')
    
    # Add annotations for the key points
    plt.annotate(f'Low Waste\n{min_waste["total_over_pct"]:.1f}% waste\n{min_waste["under_pct"]:.1f}% under', 
                xy=(min_waste['total_over_pct'], min_waste['under_pct']),
                xytext=(min_waste['total_over_pct']-5, min_waste['under_pct']+2),
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.annotate(f'Low Under\n{min_under["total_over_pct"]:.1f}% waste\n{min_under["under_pct"]:.1f}% under', 
                xy=(min_under['total_over_pct'], min_under['under_pct']),
                xytext=(min_under['total_over_pct']+3, min_under['under_pct']+1),
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.annotate(f'Balanced\n{min_business["total_over_pct"]:.1f}% waste\n{min_business["under_pct"]:.1f}% under', 
                xy=(min_business['total_over_pct'], min_business['under_pct']),
                xytext=(min_business['total_over_pct']+5, min_business['under_pct']+1),
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('Waste (Total Over-allocation %)', fontsize=14)
    plt.ylabel('Underallocation (%)', fontsize=14)
    plt.title('Pareto Frontier: Memory Allocation Trade-offs\nThree Key Configurations', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add subtitle with business context
    plt.figtext(0.5, 0.02, 
                'Green: Minimize cost (low waste) • Blue: Minimize risk (low underallocation) • Red: Optimal balance',
                ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = config.PROJECT_ROOT / "artifacts" / "pareto" / "plots" / "pareto_focused_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nFocused Pareto plot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'low_waste': min_waste,
        'low_underallocation': min_under,
        'balanced': min_business
    }


def create_and_save_models(key_points):
    """Create and save model pkl files for the three key configurations."""
    config = Config()
    
    print("\nCreating model pkl files for the three key configurations...")
    
    # Load the base trained model
    print("Loading base trained model...")
    qe_model_path = config.MODELS_DIR / "qe_regression.pkl"
    
    # Try to load as DeployableModel first, fallback to old format
    try:
        base_deployable = joblib.load(qe_model_path)
        if isinstance(base_deployable, DeployableModel):
            print("Base model is in DeployableModel format")
            base_model = base_deployable.model
            preprocessor = base_deployable.preprocessor
        else:
            raise ValueError("Not a DeployableModel")
    except Exception:
        # Load as old format and create preprocessor
        print("Loading base model from old format...")
        model_data = joblib.load(qe_model_path)
        base_model = model_data['model']
        features = model_data.get('features', [])
        
        # Create preprocessor (reconstruct from training data)
        print("Reconstructing preprocessor...")
        X_train = pd.read_pickle(config.X_TRAIN_PATH)
        preprocessor = ModelPreprocessor(
            categorical_features=config.CATEGORICAL_FEATURES,
            numerical_features=config.NUMERICAL_FEATURES,
            target_column=config.TARGET_COLUMN,
            drop_columns=config.COLUMNS_TO_DROP
        )
        preprocessor.fit(X_train)
    
    print(f"Base model: α={base_model.alpha:.3f}, safety={base_model.safety:.3f}")
    
    # Create artifacts subfolder for the three models
    models_subfolder = config.PROJECT_ROOT / "artifacts" / "pareto" / "models"
    models_subfolder.mkdir(exist_ok=True)
    
    # Extract base model parameters
    gb_params = base_model.gb.get_params()
    
    model_configs = [
        ('low_waste', key_points['low_waste'], 'green'),
        ('low_underallocation', key_points['low_underallocation'], 'blue'), 
        ('balanced', key_points['balanced'], 'red')
    ]
    
    saved_models = {}
    
    for model_name, point, color in model_configs:
        print(f"\nCreating {model_name} model...")
        print(f"  α={point['alpha']:.3f}, safety={point['safety']:.3f}")
        print(f"  Performance: {point['total_over_pct']:.1f}% waste, {point['under_pct']:.1f}% underallocation")
        
        # Create new model with the specific alpha and safety parameters
        new_model = QuantileEnsemblePredictor(
            alpha=point['alpha'],
            safety=point['safety'],
            gb_params={k: v for k, v in gb_params.items() 
                      if k not in ['alpha', 'random_state']},
            xgb_params={
                'n_estimators': 637,  # From trained model
                'max_depth': 4,       # From trained model  
                'learning_rate': 0.06799234025497858,  # From trained model
                'n_jobs': 1,
            },
            random_state=42
        )
        
        # Copy the essential attributes from the base model
        new_model.columns = base_model.columns
        new_model.gb = copy.deepcopy(base_model.gb)
        new_model.xgb = copy.deepcopy(base_model.xgb)
        
        # Update the alpha and safety in the new model
        new_model.alpha = point['alpha']
        new_model.safety = point['safety']
        
        # Create deployable model wrapper in the correct format
        deployable_model = DeployableModel(
            model=new_model,
            model_type='quantile_ensemble',
            task_type='regression',
            preprocessor=preprocessor,
            bin_edges=None,  # Regression model doesn't need bin edges
            metadata={
                'alpha': point['alpha'],
                'safety': point['safety'],
                'waste_pct': point['total_over_pct'],
                'underallocation_pct': point['under_pct'],
                'business_score': point['business_score'],
                'description': f'{model_name.replace("_", " ").title()} configuration from Pareto frontier analysis',
                'pareto_configuration': True,
                'training_timestamp': str(pd.Timestamp.now())
            }
        )
        
        # Save the deployable model
        model_path = models_subfolder / f"qe_{model_name}.pkl"
        deployable_model.save(model_path)
        print(f"  Saved to: {model_path}")
        
        # Test loading to verify it works
        try:
            test_model = DeployableModel.load(model_path)
            info = test_model.get_model_info()
            print(f"  ✓ Verified: {info['model_type']}, {info['task_type']}")
        except Exception as e:
            print(f"  ✗ Error loading: {e}")
            continue
        
        saved_models[model_name] = {
            'path': model_path,
            'config': {
                'alpha': point['alpha'],
                'safety': point['safety'],
                'waste_pct': point['total_over_pct'],
                'underallocation_pct': point['under_pct'],
                'business_score': point['business_score'],
                'description': f'{model_name.replace("_", " ").title()} configuration from Pareto frontier analysis'
            }
        }
    
    # Create a summary file
    summary_path = models_subfolder / "models_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Pareto Frontier Model Configurations\n")
        f.write("=" * 50 + "\n\n")
        f.write("Three key model configurations exported from Pareto frontier analysis:\n\n")
        
        for model_name, info in saved_models.items():
            config = info['config']
            f.write(f"{model_name.replace('_', ' ').title()}:\n")
            f.write(f"  File: {info['path'].name}\n")
            f.write(f"  Alpha: {config['alpha']:.3f}\n")
            f.write(f"  Safety: {config['safety']:.3f}\n")
            f.write(f"  Waste: {config['waste_pct']:.2f}%\n")
            f.write(f"  Underallocation: {config['underallocation_pct']:.2f}%\n")
            f.write(f"  Business Score: {config['business_score']:.2f}\n")
            f.write(f"  Description: {config['description']}\n\n")
        
        f.write("Usage:\n")
        f.write("  from resource_prediction.models import load_model\n")
        f.write("  model = load_model('path/to/model.pkl')\n")
        f.write("  predictions = model.predict(raw_dataframe)\n")
        f.write("  info = model.get_model_info()\n")
    
    print(f"\nModel summary saved to: {summary_path}")
    print(f"\nAll models saved in: {models_subfolder}")
    
    return saved_models


def main():
    """Main function to create the focused plot and export models."""
    print("Creating focused Pareto frontier plot and exporting key model configurations...")
    print("=" * 80)
    
    # Create the focused plot
    key_points = create_focused_pareto_plot()
    
    # Create and save the three key models
    saved_models = create_and_save_models(key_points)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("✓ Created focused Pareto frontier plot highlighting 3 key points")
    print("✓ Exported 3 model pkl files with optimal configurations")
    print("\nFiles created:")
    print("- pareto_focused_plot.png (visualization)")
    for model_name, info in saved_models.items():
        print(f"- {info['path'].name} ({model_name.replace('_', ' ')} configuration)")
    print("- models_summary.txt (configuration details)")


if __name__ == "__main__":
    main()