"""Plotting utilities for PeptideLocator2 experiment results."""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_experiments(target: str, results_dir: str = "results") -> list:
    """Load all experiments for a given target.

    Only keeps the latest experiment per condition (loss_type + downsampling).

    Args:
        target: 'sites' or 'peptides'
        results_dir: Directory containing result CSVs and YAMLs

    Returns:
        List of experiment dicts with metrics
    """
    all_experiments = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith(".yml") or not fname.startswith(target):
            continue
        exp_num = int(fname.replace(f"{target}-", "").replace(".yml", ""))
        meta = yaml.safe_load(open(os.path.join(results_dir, fname)))
        if "loss_type" not in meta:
            continue
        csv_path = os.path.join(results_dir, f"results-{target}-predictor-{exp_num}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        key = (meta["loss_type"], meta["downsampling"])
        entry = {
            "exp_num": exp_num,
            "loss_type": meta["loss_type"],
            "downsampling": meta["downsampling"],
            "mcc_mean": df["mcc"].mean(),
            "mcc_std": df["mcc"].std(),
            "precision_mean": df["precision_weighted"].mean(),
            "precision_std": df["precision_weighted"].std(),
            "recall_mean": df["recall_macro"].mean(),
            "recall_std": df["recall_macro"].std(),
            "valid_mcc": df["valid_mcc"].dropna() if "valid_mcc" in df.columns else None,
            "epoch": df["epoch"].dropna() if "epoch" in df.columns else None,
        }
        if key not in all_experiments or exp_num > all_experiments[key]["exp_num"]:
            all_experiments[key] = entry
    return list(all_experiments.values())


def plot_bar_comparison(experiments: list, target: str, plots_dir: str = "plots") -> None:
    """Bar chart comparing MCC across conditions.

    Args:
        experiments: List of experiment dicts from load_experiments()
        target: 'sites' or 'peptides'
        plots_dir: Directory to save the plot
    """
    labels = [
        f"{e['loss_type'].upper()}\n{'Downsample' if e['downsampling'] == 'yes' else 'No Downsample'}"
        for e in experiments
    ]
    mccs = [e["mcc_mean"] for e in experiments]
    stds = [e["mcc_std"] for e in experiments]
    colors = ["#2196F3" if e["loss_type"] == "focal" else "#FF5722" for e in experiments]
    alphas = [1.0 if e["downsampling"] == "no" else 0.6 for e in experiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, mccs, yerr=stds, capsize=5, color=colors,
                  alpha=1.0, edgecolor="black", linewidth=0.8)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title(f"MCC by Loss & Sampling Strategy\n({target.capitalize()} Predictor)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    focal_patch = mpatches.Patch(color="#2196F3", label="Focal Loss")
    bce_patch = mpatches.Patch(color="#FF5722", label="BCE Loss")
    ax.legend(handles=[focal_patch, bce_patch])

    caption = (
        f"Figure 1: MCC (mean ± std across 5 seeds × 5 folds) for each loss and sampling "
        f"condition on the {target} predictor. Higher is better."
    )
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color="gray",
             wrap=True, transform=fig.transFigure)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"mcc_comparison_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_learning_curves(experiments: list, target: str, plots_dir: str = "plots") -> None:
    """Validation MCC vs epoch for each condition.

    Args:
        experiments: List of experiment dicts from load_experiments()
        target: 'sites' or 'peptides'
        plots_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "focal_no": "#2196F3", "focal_yes": "#64B5F6",
        "bce_no": "#FF5722", "bce_yes": "#FFAB91",
    }

    for e in experiments:
        if e["valid_mcc"] is None or e["epoch"] is None:
            continue
        key = f"{e['loss_type']}_{e['downsampling']}"
        label = f"{e['loss_type'].upper()} {'+ Downsample' if e['downsampling'] == 'yes' else 'No Downsample'}"
        df_tmp = pd.DataFrame({"epoch": e["epoch"], "valid_mcc": e["valid_mcc"]})
        grouped = df_tmp.groupby("epoch")["valid_mcc"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["epoch"] + 1, grouped["mean"], label=label,
                color=colors.get(key, "gray"), linewidth=2)
        ax.fill_between(grouped["epoch"] + 1,
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.2, color=colors.get(key, "gray"))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation MCC", fontsize=12)
    ax.set_title(f"Learning Curves\n({target.capitalize()} Predictor)", fontsize=13)
    ax.legend()
    caption = (
        f"Figure 2: Validation MCC per epoch (mean ± std) for each condition on the "
        f"{target} predictor. Shaded area shows standard deviation across folds and seeds."
    )
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color="gray",
             wrap=True, transform=fig.transFigure)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"learning_curves_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_precision_recall(experiments: list, target: str, plots_dir: str = "plots") -> None:
    """Precision vs Recall scatter for each condition.

    Args:
        experiments: List of experiment dicts from load_experiments()
        target: 'sites' or 'peptides'
        plots_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"focal": "#2196F3", "bce": "#FF5722"}
    markers = {"no": "o", "yes": "^"}

    for e in experiments:
        color = colors.get(e["loss_type"], "gray")
        marker = markers.get(e["downsampling"], "o")
        label = f"{e['loss_type'].upper()} {'+ Downsample' if e['downsampling'] == 'yes' else 'No Downsample'}"
        ax.errorbar(e["recall_mean"], e["precision_mean"],
                    xerr=e["recall_std"], yerr=e["precision_std"],
                    fmt=marker, color=color, markersize=10,
                    capsize=4, label=label, linewidth=1.5)

    ax.set_xlabel("Recall (macro)", fontsize=12)
    ax.set_ylabel("Precision (weighted)", fontsize=12)
    ax.set_title(f"Precision vs Recall\n({target.capitalize()} Predictor)", fontsize=13)
    ax.legend()
    caption = (
        f"Figure 3: Precision vs Recall (mean ± std) for each condition on the {target} "
        f"predictor. Circle = no undersampling, triangle = undersampling."
    )
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color="gray",
             wrap=True, transform=fig.transFigure)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, f"precision_recall_{target}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_all(targets: list = None, results_dir: str = "results", plots_dir: str = "plots") -> None:
    """Generate all plots for all targets.

    Args:
        targets: List of targets, default ['sites', 'peptides']
        results_dir: Directory with result files
        plots_dir: Directory to save plots
    """
    if targets is None:
        targets = ["sites", "peptides"]

    for target in targets:
        experiments = load_experiments(target, results_dir=results_dir)
        if not experiments:
            print(f"No experiments found for {target}")
            continue
        experiments = sorted(experiments, key=lambda x: x["exp_num"])
        print(f"\n=== {target.upper()} ===")
        for e in experiments:
            print(f"  Exp {e['exp_num']} | {e['loss_type']} | downsample={e['downsampling']} "
                  f"| MCC={e['mcc_mean']:.3f}±{e['mcc_std']:.3f}")
        plot_bar_comparison(experiments, target, plots_dir=plots_dir)
        plot_learning_curves(experiments, target, plots_dir=plots_dir)
        plot_precision_recall(experiments, target, plots_dir=plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")
