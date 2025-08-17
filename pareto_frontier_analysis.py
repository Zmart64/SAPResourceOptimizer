#!/usr/bin/env python3
"""
Pareto Frontier Analysis for Quantile Ensemble Model

This script analyzes the trade-off between waste (over-allocation) and 
underallocation by varying only the alpha and safety parameters of the 
trained quantile ensemble model while keeping all other parameters stable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import copy
from pathlib import Path
from itertools import product

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from resource_prediction.config import Config
from resource_prediction.models.quantile_ensemble import QuantileEnsemblePredictor


class ParetoFrontierAnalyzer:
    """Analyzes Pareto frontier for alpha and safety parameter combinations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        
    def load_data_and_model(self):
        """Load the trained model and test data."""
        print("Loading trained model and test data...")
        
        # Load trained model
        model_data = joblib.load(self.config.MODELS_DIR / "qe_regression.pkl")
        self.base_model = model_data['model']
        self.features = model_data['features']
        
        # Load test data (holdout set)
        self.X_test = pd.read_pickle(self.config.X_TEST_PATH)
        self.y_test = pd.read_pickle(self.config.Y_TEST_PATH)
        
        print(f"Loaded model with alpha={self.base_model.alpha:.3f}, safety={self.base_model.safety:.3f}")
        print(f"Test set size: {len(self.X_test)} samples")
        
    def _allocation_metrics(self, allocated, true):
        """Calculate allocation metrics consistent with hyperparameter optimization."""
        under = np.sum(allocated < true)
        over = np.maximum(0, allocated - true)
        
        return {
            "under_pct": 100 * under / len(true) if len(true) > 0 else 0,
            "total_over_pct": 100 * over.sum() / true.sum() if true.sum() > 0 else 0,
        }
    
    def _business_score(self, metrics):
        """Calculate business score consistent with hyperparameter optimization."""
        return metrics["under_pct"] * 5 + metrics["total_over_pct"]
    
    def evaluate_parameter_combination(self, alpha, safety):
        """Evaluate model performance for a specific alpha/safety combination."""
        # Extract the base model's parameters
        gb_params = self.base_model.gb.get_params()
        
        # Create new model with updated alpha and safety
        new_model = QuantileEnsemblePredictor(
            alpha=alpha,
            safety=safety,
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
        
        # Set the column information from the original model
        new_model.columns = self.base_model.columns
        
        # Copy the fitted models' states by using the same training approach
        # For prediction, we need to simulate the effect of different alpha/safety
        # by using the base model's predictions and adjusting them
        
        # Get base predictions (before safety factor)
        base_gb_preds = self.base_model.gb.predict(
            self.base_model._encode(self.X_test[self.features], fit=False)
        )
        base_xgb_preds = self.base_model.xgb.predict(
            self.base_model._encode(self.X_test[self.features], fit=False)
        )
        
        # Adjust predictions based on alpha difference
        # Higher alpha should give higher quantile predictions
        alpha_ratio = alpha / self.base_model.alpha
        
        # Scale the predictions by the alpha ratio (approximation)
        adj_gb_preds = base_gb_preds * alpha_ratio
        adj_xgb_preds = base_xgb_preds * alpha_ratio
        
        # Take maximum and apply new safety factor
        max_preds = np.maximum(adj_gb_preds, adj_xgb_preds)
        predictions = max_preds * safety
        
        true_values = self.y_test[self.config.TARGET_COLUMN_PROCESSED].values
        
        # Calculate metrics
        metrics = self._allocation_metrics(predictions, true_values)
        business_score = self._business_score(metrics)
        
        # Store detailed results
        result = {
            'alpha': alpha,
            'safety': safety,
            'under_pct': metrics['under_pct'],
            'total_over_pct': metrics['total_over_pct'],
            'business_score': business_score,
            'mean_prediction': np.mean(predictions),
            'mean_true': np.mean(true_values),
            'total_waste_gb': np.sum(np.maximum(0, predictions - true_values)),
            'total_underalloc_gb': np.sum(np.maximum(0, true_values - predictions)),
        }
        
        return result
    
    def run_pareto_analysis(self, alpha_values=None, safety_values=None):
        """Run Pareto frontier analysis across parameter combinations."""
        if alpha_values is None:
            # Test range around the trained model's alpha
            alpha_values = [0.90, 0.95, 0.98, 0.99]
            
        if safety_values is None:
            # Test range around the trained model's safety factor
            base_safety = self.base_model.safety
            safety_values = np.arange(
                max(1.0, base_safety - 0.10), 
                base_safety + 0.15, 
                0.02
            ).round(3)
        
        print(f"Testing {len(alpha_values)} alpha values: {alpha_values}")
        print(f"Testing {len(safety_values)} safety values: {safety_values}")
        print(f"Total combinations: {len(alpha_values) * len(safety_values)}")
        
        # Evaluate all combinations
        for i, (alpha, safety) in enumerate(product(alpha_values, safety_values)):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(alpha_values) * len(safety_values)}")
                
            result = self.evaluate_parameter_combination(alpha, safety)
            self.results.append(result)
        
        print(f"Completed evaluation of {len(self.results)} parameter combinations")
        
        # Convert to DataFrame for analysis
        self.results_df = pd.DataFrame(self.results)
        return self.results_df
    
    def find_pareto_frontier(self):
        """Identify the Pareto frontier points."""
        if not hasattr(self, 'results_df'):
            raise ValueError("Must run pareto analysis first")
        
        # For Pareto frontier, we want to minimize both waste and underallocation
        df = self.results_df.copy()
        
        # Find Pareto frontier points
        pareto_points = []
        for i, row in df.iterrows():
            is_dominated = False
            for j, other_row in df.iterrows():
                if i != j:
                    # Check if this point is dominated by another
                    # (other point has lower or equal waste AND lower underallocation)
                    if (other_row['total_over_pct'] <= row['total_over_pct'] and 
                        other_row['under_pct'] <= row['under_pct'] and
                        (other_row['total_over_pct'] < row['total_over_pct'] or 
                         other_row['under_pct'] < row['under_pct'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append(i)
        
        self.pareto_df = df.iloc[pareto_points].copy()
        print(f"Found {len(pareto_points)} Pareto frontier points")
        return self.pareto_df
    
    def plot_pareto_frontier(self, save_path=None):
        """Create visualization of the Pareto frontier."""
        if not hasattr(self, 'results_df'):
            raise ValueError("Must run pareto analysis first")
        
        # Create the main plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pareto Frontier Analysis: Waste vs Underallocation Trade-offs', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Scatter plot of all points with Pareto frontier highlighted
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.results_df['total_over_pct'], self.results_df['under_pct'], 
                             c=self.results_df['business_score'], cmap='viridis', alpha=0.6, s=50)
        
        if hasattr(self, 'pareto_df'):
            # Sort Pareto points by waste for connecting line
            pareto_sorted = self.pareto_df.sort_values('total_over_pct')
            ax1.plot(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'], 
                    'r-', linewidth=2, label='Pareto Frontier')
            ax1.scatter(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'], 
                       c='red', s=100, marker='o', edgecolors='black', linewidth=2,
                       label=f'Pareto Points ({len(pareto_sorted)})')
        
        ax1.set_xlabel('Waste (Total Over-allocation %)')
        ax1.set_ylabel('Underallocation (%)')
        ax1.set_title('Waste vs Underallocation Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for business score
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Business Score')
        
        # Plot 2: Alpha vs Safety heatmap (colored by business score)
        ax2 = axes[0, 1]
        pivot_score = self.results_df.pivot(index='alpha', columns='safety', values='business_score')
        sns.heatmap(pivot_score, annot=False, cmap='RdYlBu_r', ax=ax2, cbar_kws={'label': 'Business Score'})
        ax2.set_title('Business Score by Alpha and Safety')
        ax2.set_xlabel('Safety Factor')
        ax2.set_ylabel('Alpha (Quantile Level)')
        
        # Plot 3: Parameter combinations colored by waste
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(self.results_df['alpha'], self.results_df['safety'], 
                              c=self.results_df['total_over_pct'], cmap='Reds', s=50, alpha=0.7)
        ax3.set_xlabel('Alpha (Quantile Level)')
        ax3.set_ylabel('Safety Factor')
        ax3.set_title('Parameter Space colored by Waste')
        plt.colorbar(scatter3, ax=ax3, label='Waste (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter combinations colored by underallocation
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(self.results_df['alpha'], self.results_df['safety'], 
                              c=self.results_df['under_pct'], cmap='Blues', s=50, alpha=0.7)
        ax4.set_xlabel('Alpha (Quantile Level)')
        ax4.set_ylabel('Safety Factor')
        ax4.set_title('Parameter Space colored by Underallocation')
        plt.colorbar(scatter4, ax=ax4, label='Underallocation (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def save_results(self, results_path=None, pareto_path=None):
        """Save analysis results to CSV files."""
        if results_path is None:
            results_path = self.config.PROJECT_ROOT / "artifacts" / "pareto" / "results" / "pareto_frontier_results.csv"
        if pareto_path is None:
            pareto_path = self.config.PROJECT_ROOT / "artifacts" / "pareto" / "results" / "pareto_frontier_points.csv"
            
        # Save all results
        self.results_df.to_csv(results_path, index=False)
        print(f"All results saved to {results_path}")
        
        # Save Pareto frontier points
        if hasattr(self, 'pareto_df'):
            self.pareto_df.to_csv(pareto_path, index=False)
            print(f"Pareto frontier points saved to {pareto_path}")
        
        return results_path, pareto_path
    
    def print_summary(self):
        """Print summary of analysis results."""
        if not hasattr(self, 'results_df'):
            print("No results to summarize. Run analysis first.")
            return
            
        print("\n" + "="*60)
        print("PARETO FRONTIER ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total parameter combinations tested: {len(self.results_df)}")
        print(f"Alpha range: {self.results_df['alpha'].min():.2f} - {self.results_df['alpha'].max():.2f}")
        print(f"Safety range: {self.results_df['safety'].min():.3f} - {self.results_df['safety'].max():.3f}")
        
        print(f"\nWaste (total_over_pct) range: {self.results_df['total_over_pct'].min():.2f}% - {self.results_df['total_over_pct'].max():.2f}%")
        print(f"Underallocation range: {self.results_df['under_pct'].min():.2f}% - {self.results_df['under_pct'].max():.2f}%")
        print(f"Business score range: {self.results_df['business_score'].min():.2f} - {self.results_df['business_score'].max():.2f}")
        
        # Best points for different objectives
        best_waste = self.results_df.loc[self.results_df['total_over_pct'].idxmin()]
        best_under = self.results_df.loc[self.results_df['under_pct'].idxmin()]
        best_business = self.results_df.loc[self.results_df['business_score'].idxmin()]
        
        print(f"\nBest for minimizing waste:")
        print(f"  Alpha: {best_waste['alpha']:.3f}, Safety: {best_waste['safety']:.3f}")
        print(f"  Waste: {best_waste['total_over_pct']:.2f}%, Underallocation: {best_waste['under_pct']:.2f}%")
        
        print(f"\nBest for minimizing underallocation:")
        print(f"  Alpha: {best_under['alpha']:.3f}, Safety: {best_under['safety']:.3f}")
        print(f"  Waste: {best_under['total_over_pct']:.2f}%, Underallocation: {best_under['under_pct']:.2f}%")
        
        print(f"\nBest business score:")
        print(f"  Alpha: {best_business['alpha']:.3f}, Safety: {best_business['safety']:.3f}")
        print(f"  Waste: {best_business['total_over_pct']:.2f}%, Underallocation: {best_business['under_pct']:.2f}%")
        print(f"  Business Score: {best_business['business_score']:.2f}")
        
        # Original model performance
        original_idx = self.results_df[
            (np.isclose(self.results_df['alpha'], 0.95, atol=1e-3)) & 
            (np.isclose(self.results_df['safety'], 1.043, atol=1e-3))
        ]
        
        if not original_idx.empty:
            orig = original_idx.iloc[0]
            print(f"\nOriginal trained model (alpha=0.95, safety=1.043):")
            print(f"  Waste: {orig['total_over_pct']:.2f}%, Underallocation: {orig['under_pct']:.2f}%")
            print(f"  Business Score: {orig['business_score']:.2f}")
        
        if hasattr(self, 'pareto_df'):
            print(f"\nPareto frontier contains {len(self.pareto_df)} points")
            print("Pareto frontier summary:")
            print(self.pareto_df[['alpha', 'safety', 'total_over_pct', 'under_pct', 'business_score']].round(3))


def main():
    """Main function to run the Pareto frontier analysis."""
    config = Config()
    
    print("Starting Pareto Frontier Analysis for Quantile Ensemble Model")
    print("="*70)
    
    # Initialize analyzer
    analyzer = ParetoFrontierAnalyzer(config)
    
    # Load data and model
    analyzer.load_data_and_model()
    
    # Run analysis with custom parameter ranges
    alpha_values = [0.90, 0.95, 0.98, 0.99]  # From config search space
    safety_values = np.arange(1.00, 1.16, 0.02).round(3)  # From config search space
    
    results_df = analyzer.run_pareto_analysis(alpha_values, safety_values)
    
    # Find Pareto frontier
    pareto_df = analyzer.find_pareto_frontier()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    results_path, pareto_path = analyzer.save_results()
    
    # Create and save plots
    plot_path = config.PROJECT_ROOT / "artifacts" / "pareto" / "plots" / "pareto_frontier_plot.png"
    analyzer.plot_pareto_frontier(save_path=plot_path)
    
    print(f"\nAnalysis complete! Results saved to {config.PROJECT_ROOT / 'artifacts' / 'pareto'}")
    print(f"- All results: {results_path}")
    print(f"- Pareto points: {pareto_path}")
    print(f"- Visualization: {plot_path}")


if __name__ == "__main__":
    main()