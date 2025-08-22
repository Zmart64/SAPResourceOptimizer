"""
Plotting utilities for Pareto frontier analysis.
"""
import matplotlib.pyplot as plt


def plot_frontier(all_points_df, pareto_df, key_points, output_path):
    """Generate and save a focused Pareto frontier plot with all points shown.

    - Plot all evaluated (alpha, safety) points as light markers.
    - Overlay Pareto-optimal points in a darker tone and connect them.
    - Highlight three key operating points with distinct markers and annotations.
    """
    pareto_sorted = pareto_df.sort_values('total_over_pct')

    plt.figure(figsize=(10, 8))

    # All points (light background scatter)
    plt.scatter(
        all_points_df['total_over_pct'],
        all_points_df['under_pct'],
        c='#cccccc', s=30, alpha=0.5, label='All configurations'
    )

    # Pareto points (darker) and frontier line
    plt.scatter(
        pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
        c='#444444', s=60, alpha=0.9, label='Pareto frontier points'
    )
    plt.plot(
        pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
        color='#333333', linewidth=2.0, alpha=0.9, label='Pareto frontier'
    )

    # Highlight key points
    markers = {'low_waste': ('o', 'green'),
               'low_underallocation': ('s', 'blue'),
               'balanced': ('^', 'red')}
    for name, point in key_points.items():
        marker, color = markers[name]
        plt.scatter(
            point['total_over_pct'], point['under_pct'],
            c=color, s=200, marker=marker,
            edgecolors='black', linewidth=2,
            label=f"{name.replace('_', ' ').title()} (α={point['alpha']:.3f}, s={point['safety']:.3f})"
        )

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
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.25),
            arrowprops=dict(arrowstyle='->', color=color)
        )

    plt.xlabel('Waste (Total Over-allocation %)', fontsize=14)
    plt.ylabel('Underallocation (%)', fontsize=14)
    plt.title('Pareto Frontier: Memory Allocation Trade-offs\nConfigurations and Frontier',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Focused Pareto plot saved to: {output_path}")
