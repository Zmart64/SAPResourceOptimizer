# Pareto Frontier Analysis Results

This directory contains all outputs from the Pareto frontier analysis of the quantile ensemble model, organized into logical subdirectories:

## Directory Structure

### `plots/`
- `pareto_frontier_plot.png` - Comprehensive 4-panel visualization showing all parameter combinations and trade-offs
- `pareto_focused_plot.png` - Single focused plot highlighting the three key configurations on the Pareto frontier

### `results/`
- `pareto_frontier_results.csv` - Complete results for all 32 tested parameter combinations (alpha × safety)
- `pareto_frontier_points.csv` - The 30 Pareto-optimal points representing the best trade-offs

### `models/`
- `qe_low_waste.pkl` - Model configuration optimized for minimal waste (α=0.90, safety=1.00)
- `qe_low_underallocation.pkl` - Model configuration optimized for minimal underallocation (α=0.99, safety=1.14)
- `qe_balanced.pkl` - Balanced configuration optimizing business score (α=0.98, safety=1.04)
- `models_summary.txt` - Detailed summary of all three model configurations

## Key Findings

The analysis explores trade-offs between:
- **Waste (total_over_pct)**: Percentage of total memory that is over-allocated
- **Underallocation (under_pct)**: Percentage of jobs where allocated < true memory needed

### Three Key Configurations:
1. **Low Waste** (Green): 29.8% waste, 13.0% underallocation - Minimizes cost
2. **Low Underallocation** (Blue): 61.2% waste, 0.7% underallocation - Minimizes risk  
3. **Balanced** (Red): 45.8% waste, 2.6% underallocation - Optimal business score

## Usage

Load any of the three optimized models:
```python
import joblib
model_data = joblib.load('artifacts/pareto/models/qe_balanced.pkl')
model = model_data['model']
features = model_data['features']
config = model_data['config']
```

Each model contains all trained components and is ready for immediate deployment.