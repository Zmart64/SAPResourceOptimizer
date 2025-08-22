"""
Command-line interface for Pareto tools.
"""
import argparse
from resource_prediction.config import Config
from resource_prediction.pareto.core import load_frontier, get_key_points, generate_frontier
from resource_prediction.pareto.plot import plot_frontier
from resource_prediction.pareto.export_models import save_models


def main():
    parser = argparse.ArgumentParser(prog='pareto', description='Pareto frontier utilities')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('analyze', help='Print analysis of Pareto results')
    p_plot = sub.add_parser('plot', help='Generate focused Pareto plot')
    p_sweep = sub.add_parser('sweep', help='Retrain QE for alpha grid and regenerate frontier CSV')
    # Optional grids for sweep
    p_sweep.add_argument('--alpha-grid', type=str, default=None,
                         help='Comma-separated list of alpha values (e.g., 0.9,0.95,0.98)')
    p_sweep.add_argument('--alpha-start', type=float, default=0.85)
    p_sweep.add_argument('--alpha-end', type=float, default=0.99)
    p_sweep.add_argument('--alpha-steps', type=int, default=15)
    p_sweep.add_argument('--safety-start', type=float, default=1.00)
    p_sweep.add_argument('--safety-end', type=float, default=1.15)
    p_sweep.add_argument('--safety-steps', type=int, default=13)
    sub.add_parser('export', help='Export models for key Pareto configurations')
    sub.add_parser('all', help='Run analyze, plot, and export in sequence')

    args = parser.parse_args()
    config = Config()
    results_dir = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'results'
    frontier_csv = results_dir / 'pareto_frontier_points.csv'

    if args.command == 'analyze':
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        print(kp)
    elif args.command == 'plot':
        points_csv = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'results' / 'pareto_all_points.csv'
        all_df = load_frontier(points_csv)
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        out = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'plots' / 'pareto_focused_plot.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_frontier(all_df, df, kp, out)
    elif args.command == 'sweep':
        print("Generating Pareto frontier by retraining QE across alphas and safeties...")
        import numpy as np
        if args.alpha_grid:
            alphas = [float(x) for x in args.alpha_grid.split(',')]
        else:
            alphas = np.round(np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps), 3)
        safeties = np.round(np.linspace(args.safety_start, args.safety_end, args.safety_steps), 3)
        path = generate_frontier(config, alphas=alphas, safeties=safeties)
        print(f"Frontier saved to: {path}")
    elif args.command == 'export':
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        saved = save_models(kp, config)
        print(f"Exported models: {saved}")
    elif args.command == 'all':
        # sweep first to regenerate points with a denser default grid
        print("Sweeping and regenerating frontier points...")
        import numpy as np
        alphas = np.round(np.linspace(0.85, 0.99, 15), 3)
        safeties = np.round(np.linspace(1.00, 1.15, 13), 3)
        frontier_csv = generate_frontier(config, alphas=alphas, safeties=safeties)
        # analyze
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        print(kp)
        # plot
        out = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'plots' / 'pareto_focused_plot.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        points_csv = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'results' / 'pareto_all_points.csv'
        all_df = load_frontier(points_csv)
        plot_frontier(all_df, df, kp, out)
        # export
        saved = save_models(kp, config)
        print(f"Exported: {saved}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
