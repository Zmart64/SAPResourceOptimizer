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

    # Compact figure suitable for ACM single-column width (~3.3in)
    plt.figure(figsize=(3.4, 2.6), dpi=400)

    # All points (light background scatter)
    plt.scatter(
        all_points_df['total_over_pct'],
        all_points_df['under_pct'],
        c='#cccccc', s=12, alpha=0.55, label='_nolegend_'
    )

    # Pareto points (darker) and frontier line
    plt.scatter(
        pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
        c='#555555', s=28, alpha=0.9, label='_nolegend_'
    )
    plt.plot(
        pareto_sorted['total_over_pct'], pareto_sorted['under_pct'],
        color='#333333', linewidth=1.6, alpha=0.9, label='Pareto frontier'
    )

    # Highlight key points
    markers = {'low_waste': ('o', 'green'),
               'low_underallocation': ('s', 'blue'),
               'balanced': ('^', 'red')}
    for name, point in key_points.items():
        marker, color = markers[name]
        plt.scatter(
            point['total_over_pct'], point['under_pct'],
            c=color, s=120, marker=marker,
            edgecolors='black', linewidth=0.9, zorder=5,
            label=f"{name.replace('_', ' ').title()} (α={point['alpha']:.2f}, s={point['safety']:.2f})"
        )

    # Note: Previously we added callouts (bubbles) for key points.
    # Per request, remove external annotations and keep the plot clean.

    # Axes styling and labels for small figure
    ax = plt.gca()
    ax.set_xlabel('Waste (Total Over-allocation %)', fontsize=9)
    ax.set_ylabel('Underallocation (%)', fontsize=9)
    ax.set_title('Pareto Frontier (Waste vs Underallocation)', fontsize=9, pad=4)
    # Compute data-driven limits: minimize left whitespace and avoid clipping bottom markers
    max_x = max(
        float(all_points_df['total_over_pct'].max()),
        float(pareto_sorted['total_over_pct'].max()),
        max(float(p['total_over_pct']) for p in key_points.values())
    )
    min_x = min(
        float(all_points_df['total_over_pct'].min()),
        float(pareto_sorted['total_over_pct'].min()),
        min(float(p['total_over_pct']) for p in key_points.values())
    )
    max_y = max(
        float(all_points_df['under_pct'].max()),
        float(pareto_sorted['under_pct'].max()),
        max(float(p['under_pct']) for p in key_points.values())
    )
    # Pad: almost no left whitespace, a bit on right/top; extra space below to avoid axis overlap
    x_range = max(1e-6, max_x - min_x)
    left = max(1e-6, min_x - 0.05 * x_range)
    right = max_x + 0.02 * x_range
    y_range = max(1e-6, max_y - 0.0)
    bottom = -0.05 * y_range
    top = max_y * 1.05
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Place legend inside, upper-right corner, compact
    leg = ax.legend(
        loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1,
        fontsize=7.8, frameon=True, fancybox=True, framealpha=0.92,
        borderpad=0.25, handletextpad=0.3, markerscale=0.6
    )
    leg.get_frame().set_linewidth(0.7)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    plt.tight_layout(pad=0.1)

    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Focused Pareto plot saved to: {output_path}")
