"""
Command-line interface for Pareto tools.
"""
import argparse
from resource_prediction.config import Config
from resource_prediction.pareto.core import load_frontier, get_key_points
from resource_prediction.pareto.plot import plot_frontier
from resource_prediction.pareto.export_models import save_models


def main():
    parser = argparse.ArgumentParser(prog='pareto', description='Pareto frontier utilities')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('analyze', help='Print analysis of Pareto results')
    sub.add_parser('plot', help='Generate focused Pareto plot')
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
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        out = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'plots' / 'pareto_focused_plot.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_frontier(df, kp, out)
    elif args.command == 'export':
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        saved = save_models(kp, config)
        print(f"Exported models: {saved}")
    elif args.command == 'all':
        # analyze
        df = load_frontier(frontier_csv)
        kp = get_key_points(df)
        print(kp)
        # plot
        out = config.PROJECT_ROOT / 'artifacts' / 'pareto' / 'plots' / 'pareto_focused_plot.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_frontier(df, kp, out)
        # export
        saved = save_models(kp, config)
        print(f"Exported: {saved}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
