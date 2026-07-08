"""Command-line interface for PeptideLocator2."""

import argparse
import sys


def cmd_train(args):
    from .train import run_experiment
    run_experiment(
        target=args.target,
        data_path=args.data,
        results_dir=args.results_dir,
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        alpha_weight=args.alpha,
        gamma=args.gamma,
        loss_type=args.loss,
        downsample=args.downsample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_seeds=args.seeds,
        num_folds=args.folds,
    )


def cmd_plot(args):
    from .plot import plot_all
    targets = args.target if args.target else ["sites", "peptides"]
    plot_all(targets=targets, results_dir=args.results_dir, plots_dir=args.plots_dir)


def main():
    parser = argparse.ArgumentParser(
        prog="peptidelocator",
        description="PeptideLocator2: predict cleavage sites and peptide regions in protein sequences.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Run a cross-validation experiment")
    train_parser.add_argument("target", choices=["sites", "peptides"], help="Prediction target")
    train_parser.add_argument("--data", default="processed-data/peptide-partitions.pqt",
                              help="Path to partitions parquet file")
    train_parser.add_argument("--results-dir", default="results", help="Directory to save results")
    train_parser.add_argument("--layers", type=int, default=2, help="Number of MLP hidden layers")
    train_parser.add_argument("--hidden-size", type=int, default=320, help="MLP hidden size")
    train_parser.add_argument("--alpha", type=float, default=0.1, help="Minority class weight")
    train_parser.add_argument("--gamma", type=float, default=3.0, help="Focal loss gamma")
    train_parser.add_argument("--loss", choices=["focal", "bce"], default="focal", help="Loss function")
    train_parser.add_argument("--downsample", action="store_true", help="Undersample majority class")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seeds", type=int, default=5)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.set_defaults(func=cmd_train)

    # --- plot ---
    plot_parser = subparsers.add_parser("plot", help="Generate result plots")
    plot_parser.add_argument("--target", nargs="+", choices=["sites", "peptides"],
                             help="Targets to plot (default: both)")
    plot_parser.add_argument("--results-dir", default="results")
    plot_parser.add_argument("--plots-dir", default="plots")
    plot_parser.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
