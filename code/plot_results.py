import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_experiments(target):
    """Load all experiments for a given target (sites or peptides).
    Only keeps the latest experiment per condition (loss_type + downsampling)."""
    all_experiments = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".yml") or not fname.startswith(target):
            continue
        exp_num = int(fname.replace(f"{target}-", "").replace(".yml", ""))
        meta = yaml.safe_load(open(os.path.join(RESULTS_DIR, fname)))
        if "loss_type" not in meta:
            continue  # skip old experiments without loss_type
        csv_path = os.path.join(RESULTS_DIR, f"results-{target}-predictor-{exp_num}.csv")
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
        # Keep only the latest experiment per condition
        if key not in all_experiments or exp_num > all_experiments[key]["exp_num"]:
            all_experiments[key] = entry
    return list(all_experiments.values())


def plot_bar_comparison(experiments, target):
    """Bar chart comparing MCC across conditions."""
    labels = [f"{e['loss_type'].upper()}\n{'Downsample' if e['downsampling'] == 'yes' else 'No Downsample'}"
              for e in experiments]
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

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"mcc_comparison_{target}.png"), dpi=150)
    plt.close()
    print(f"Saved: plots/mcc_comparison_{target}.png")


def plot_learning_curves(experiments, target):
    """Validation MCC vs epoch for each condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"focal_no": "#2196F3", "focal_yes": "#64B5F6",
              "bce_no": "#FF5722", "bce_yes": "#FFAB91"}

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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"learning_curves_{target}.png"), dpi=150)
    plt.close()
    print(f"Saved: plots/learning_curves_{target}.png")


def plot_precision_recall(experiments, target):
    """Precision vs Recall scatter for each condition."""
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
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"precision_recall_{target}.png"), dpi=150)
    plt.close()
    print(f"Saved: plots/precision_recall_{target}.png")


if __name__ == "__main__":
    for target in ["sites", "peptides"]:
        experiments = load_experiments(target)
        if not experiments:
            print(f"No experiments found for {target}")
            continue
        experiments = sorted(experiments, key=lambda x: x["exp_num"])
        print(f"\n=== {target.upper()} ===")
        for e in experiments:
            print(f"  Exp {e['exp_num']} | {e['loss_type']} | downsample={e['downsampling']} | MCC={e['mcc_mean']:.3f}±{e['mcc_std']:.3f}")
        plot_bar_comparison(experiments, target)
        plot_learning_curves(experiments, target)
        plot_precision_recall(experiments, target)
    print("\nAll plots saved to plots/")
