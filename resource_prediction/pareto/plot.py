"""
Plotting utilities for Pareto frontier analysis.
"""
import matplotlib.pyplot as plt

def plot_frontier(pareto_df, key_points, output_path):
    """Generate and save a focused Pareto frontier plot with key points."""
    # Sort Pareto points by waste for connecting line
    pareto_sorted = pareto_df.sort_values('total_over_pct')

    plt.figure(figsize=(10, 8))
    # Plot all Pareto frontier points
    plt.scatter(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
                c='lightgray', s=60, alpha=0.7, label='Pareto Frontier Points')
    # Plot the Pareto frontier line
    plt.plot(pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
             'gray', linewidth=2, alpha=0.8, label='Pareto Frontier')

    # Highlight key points
    markers = {'low_waste': ('o', 'green'),
               'low_underallocation': ('s', 'blue'),
               'balanced': ('^', 'red')}
    for name, point in key_points.items():
        marker, color = markers[name]
        plt.scatter(point['total_over_pct'], point['under_pct'],
                    c=color, s=200, marker=marker,
                    edgecolors='black', linewidth=2,
                    label=f"{name.replace('_', ' ').title()} (α={point['alpha']:.2f}, s={point['safety']:.2f})")

    # Annotations
    offsets = {'low_waste': (-5, 2), 'low_underallocation': (3, 1), 'balanced': (5, 1)}
    for name, point in key_points.items():
        dx, dy = offsets[name]
        color = markers[name][1]
        plt.annotate(
            f"{name.replace('_', ' ').title()}\n{point['total_over_pct']:.1f}% waste\n{point['under_pct']:.1f}% under",
            xy=(point['total_over_pct'], point['under_pct']),
            xytext=(point['total_over_pct'] + dx, point['under_pct'] + dy),
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
            arrowprops=dict(arrowstyle='->', color=color)
        )

    plt.xlabel('Waste (Total Over-allocation %)', fontsize=14)
    plt.ylabel('Underallocation (%)', fontsize=14)
    plt.title('Pareto Frontier: Memory Allocation Trade-offs\nThree Key Configurations',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.figtext(0.5, 0.02,
                'Green: Minimize cost (low waste) • Blue: Minimize risk (low underallocation) • Red: Optimal balance',
                ha='center', fontsize=12, style='italic')
    plt.tight_layout()

    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Focused Pareto plot saved to: {output_path}")
