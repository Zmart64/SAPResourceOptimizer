#!/usr/bin/env python3
"""
Summary script to analyze and display the Pareto frontier results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from resource_prediction.config import Config

def analyze_results():
    """Analyze and summarize the Pareto frontier results."""
    config = Config()
    
    # Load results
    results_path = config.OUTPUT_DIR / "pareto_frontier_results.csv"
    pareto_path = config.OUTPUT_DIR / "pareto_frontier_points.csv"
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    results_df = pd.read_csv(results_path)
    
    print("PARETO FRONTIER ANALYSIS - DETAILED RESULTS")
    print("=" * 60)
    
    print(f"\nTotal combinations tested: {len(results_df)}")
    print(f"Parameter ranges:")
    print(f"  Alpha: {results_df['alpha'].min()} - {results_df['alpha'].max()}")
    print(f"  Safety: {results_df['safety'].min():.3f} - {results_df['safety'].max():.3f}")
    
    print(f"\nMetrics ranges:")
    print(f"  Waste (total_over_pct): {results_df['total_over_pct'].min():.2f}% - {results_df['total_over_pct'].max():.2f}%")
    print(f"  Underallocation (under_pct): {results_df['under_pct'].min():.2f}% - {results_df['under_pct'].max():.2f}%")
    print(f"  Business score: {results_df['business_score'].min():.2f} - {results_df['business_score'].max():.2f}")
    
    # Find interesting points
    min_waste = results_df.loc[results_df['total_over_pct'].idxmin()]
    min_under = results_df.loc[results_df['under_pct'].idxmin()]
    min_business = results_df.loc[results_df['business_score'].idxmin()]
    
    print(f"\nKEY FINDINGS:")
    print(f"1. Minimum Waste Configuration:")
    print(f"   Alpha: {min_waste['alpha']:.3f}, Safety: {min_waste['safety']:.3f}")
    print(f"   → Waste: {min_waste['total_over_pct']:.2f}%, Underallocation: {min_waste['under_pct']:.2f}%")
    print(f"   → Trade-off: Lower waste but higher underallocation risk")
    
    print(f"\n2. Minimum Underallocation Configuration:")
    print(f"   Alpha: {min_under['alpha']:.3f}, Safety: {min_under['safety']:.3f}")
    print(f"   → Waste: {min_under['total_over_pct']:.2f}%, Underallocation: {min_under['under_pct']:.2f}%")
    print(f"   → Trade-off: Lower risk but much higher waste")
    
    print(f"\n3. Best Business Score (balanced):")
    print(f"   Alpha: {min_business['alpha']:.3f}, Safety: {min_business['safety']:.3f}")
    print(f"   → Waste: {min_business['total_over_pct']:.2f}%, Underallocation: {min_business['under_pct']:.2f}%")
    print(f"   → Business Score: {min_business['business_score']:.2f}")
    
    # Find original model performance
    original_alpha = 0.95
    original_safety = 1.043
    
    original_result = results_df[
        (np.abs(results_df['alpha'] - original_alpha) < 0.01) & 
        (np.abs(results_df['safety'] - original_safety) < 0.01)
    ]
    
    if len(original_result) > 0:
        orig = original_result.iloc[0]
        print(f"\n4. Original Trained Model Performance:")
        print(f"   Alpha: {orig['alpha']:.3f}, Safety: {orig['safety']:.3f}")
        print(f"   → Waste: {orig['total_over_pct']:.2f}%, Underallocation: {orig['under_pct']:.2f}%")
        print(f"   → Business Score: {orig['business_score']:.2f}")
        
        # Compare with best business score
        improvement = orig['business_score'] - min_business['business_score']
        print(f"   → Potential improvement: {improvement:.2f} business score points")
    
    # Load Pareto points if available
    if pareto_path.exists():
        pareto_df = pd.read_csv(pareto_path)
        print(f"\nPARETO FRONTIER ANALYSIS:")
        print(f"  Found {len(pareto_df)} Pareto-optimal points")
        print(f"  These represent the best trade-offs between waste and underallocation")
        
        # Show some representative Pareto points
        print(f"\nRepresentative Pareto Points:")
        for i, idx in enumerate([0, len(pareto_df)//4, len(pareto_df)//2, 3*len(pareto_df)//4, -1]):
            point = pareto_df.iloc[idx]
            print(f"  {i+1}. Alpha: {point['alpha']:.3f}, Safety: {point['safety']:.3f}")
            print(f"     → Waste: {point['total_over_pct']:.2f}%, Under: {point['under_pct']:.2f}%")
    
    print(f"\nWASTE vs UNDERALLOCATION INSIGHTS:")
    print(f"• Lower alpha (0.90) = Lower quantile predictions = Less conservative = More waste, less underallocation")
    print(f"• Higher alpha (0.99) = Higher quantile predictions = More conservative = Less waste, more underallocation")  
    print(f"• Lower safety factor = Less additional buffer = More waste, less underallocation")
    print(f"• Higher safety factor = More additional buffer = Less waste, more underallocation")
    print(f"• Business score weights underallocation 5x more than waste (under_pct * 5 + total_over_pct)")
    
    print(f"\nRECOMMENDATIONS:")
    if len(original_result) > 0 and improvement > 1:
        print(f"• Current model could be improved by adjusting to alpha={min_business['alpha']:.3f}, safety={min_business['safety']:.3f}")
        print(f"• This would reduce business score by {improvement:.2f} points")
    print(f"• For cost-sensitive applications: Use alpha=0.90, safety=1.00 (minimize waste)")
    print(f"• For failure-sensitive applications: Use alpha=0.99, safety=1.14 (minimize underallocation)")
    print(f"• For balanced approach: Use alpha={min_business['alpha']:.3f}, safety={min_business['safety']:.3f}")

def main():
    analyze_results()

if __name__ == "__main__":
    main()