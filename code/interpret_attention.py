"""
Aggregate attention visualisation around cleavage sites across all 6 ESM2-8M layers.

For each layer, averages the attention pattern (across all heads) centred on
cleavage site residues across the whole dataset. Saves a 2x3 grid of heatmaps.

Each heatmap shows: for a cleavage site at position 0, how much does each
surrounding residue attend to it (and it to them) on average?

Usage:
    python code/interpret_attention.py
    python code/interpret_attention.py --window 15 --output plots/attention
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from transformers import EsmModel, EsmTokenizer


# ── Data loading ──────────────────────────────────────────────────────────────

def reconstruct_filtered_sequences():
    """Reconstruct filtered protein sequences in the same order as original-prot."""
    df = pd.read_csv("processed-data/peptide.tsv", sep="\t")
    raw_sites = df["Peptide"].map(lambda x: x.split(";"))
    sites = []
    for site_list, length in zip(raw_sites, df["Length"]):
        s = []
        for site in site_list:
            if ">" in site or "<" in site or "?" in site:
                continue
            try:
                start = int(site.split("-")[0])
                end = int(site.split("-")[1])
                if start != 1:
                    s.append(start - 1)
                if end != length:
                    s.append(end + 1)
            except Exception:
                continue
        sites.append(s)
    df["Sites"] = sites
    df = df[df["Sites"].map(len) > 0].reset_index(drop=True)
    df = df[df["Sequence"].map(len) <= 1022].reset_index(drop=True)
    return df["Sequence"].tolist()


# ── Attention extraction ───────────────────────────────────────────────────────

def extract_attention_profiles(sequences, labels_per_protein, window, device,
                               batch_size=8, use_random=False, rng=None):
    """
    For each cleavage site (or random control position), extract a centred
    attention window at each layer.

    Args:
        use_random: if True, sample random non-site positions (same count per
                    protein as real sites) instead of real cleavage sites.
        rng:        np.random.Generator for reproducibility.

    Returns:
        inbound_profiles:  {layer: np.array (2*window+1,)}
        outbound_profiles: {layer: np.array (2*window+1,)}
        n_sites: int
    """
    if rng is None:
        rng = np.random.default_rng(42)

    print("Loading ESM2-8M from HuggingFace...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        attn_implementation="eager"
    ).to(device)
    model.eval()

    W = window
    size = 2 * W + 1
    inbound_sum  = {i: np.zeros(size) for i in range(6)}
    outbound_sum = {i: np.zeros(size) for i in range(6)}
    n_sites = 0

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs   = sequences[i:i + batch_size]
            batch_labels = labels_per_protein[i:i + batch_size]

            tokens = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            outputs = model(**tokens, output_attentions=True)

            for b_idx, (seq, labels) in enumerate(zip(batch_seqs, batch_labels)):
                seq_len = len(seq)
                real_sites = np.where(labels == 1)[0]

                if use_random:
                    # Sample same number of random positions from non-site residues
                    non_sites = np.where(labels == 0)[0]
                    n_sample = min(len(real_sites), len(non_sites))
                    if n_sample == 0:
                        continue
                    positions = rng.choice(non_sites, size=n_sample, replace=False)
                else:
                    positions = real_sites
                    if len(positions) == 0:
                        continue

                for layer_idx in range(6):
                    attn = outputs.attentions[layer_idx][b_idx]
                    attn_mean = attn[:, 1:seq_len + 1, 1:seq_len + 1].mean(dim=0).cpu().numpy()

                    for site_pos in positions:
                        start = site_pos - W
                        end   = site_pos + W + 1
                        out_start = max(0, W - site_pos)
                        out_end   = out_start + min(end, seq_len) - max(start, 0)
                        seq_start = max(0, start)
                        seq_end   = min(seq_len, end)

                        inbound_sum[layer_idx][out_start:out_end] += \
                            attn_mean[seq_start:seq_end, site_pos]
                        outbound_sum[layer_idx][out_start:out_end] += \
                            attn_mean[site_pos, seq_start:seq_end]

                        n_sites += 1 if layer_idx == 0 else 0

            if i % (batch_size * 10) == 0:
                label = "random" if use_random else "real"
                print(f"  [{label}] Processed {i}/{len(sequences)} proteins")

    inbound_profiles  = {k: v / max(n_sites, 1) for k, v in inbound_sum.items()}
    outbound_profiles = {k: v / max(n_sites, 1) for k, v in outbound_sum.items()}
    return inbound_profiles, outbound_profiles, n_sites


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_attention_profiles(inbound, outbound, window, n_sites, output_dir,
                            inbound_rand=None, outbound_rand=None, n_rand=0):
    os.makedirs(output_dir, exist_ok=True)
    W = window
    x = np.arange(-W, W + 1)

    layer_names = [
        "Layer 0 — local chemistry",
        "Layer 1 — local motifs",
        "Layer 2 — short-range patterns",
        "Layer 3 — sequence context",
        "Layer 4 — regional integration",
        "Layer 5 — output compilation",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for layer_idx in range(6):
        ax = axes[layer_idx]
        ax.plot(x, inbound[layer_idx],  color="#EF5350", linewidth=2,
                label="Inbound — real sites")
        ax.plot(x, outbound[layer_idx], color="#42A5F5", linewidth=2,
                label="Outbound — real sites")
        if inbound_rand is not None:
            ax.plot(x, inbound_rand[layer_idx],  color="#EF5350", linewidth=1.5,
                    linestyle="--", alpha=0.6, label="Inbound — random")
            ax.plot(x, outbound_rand[layer_idx], color="#42A5F5", linewidth=1.5,
                    linestyle="--", alpha=0.6, label="Outbound — random")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_title(layer_names[layer_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Position relative to cleavage site", fontsize=9)
        ax.set_ylabel("Mean attention weight", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    rand_note = f" vs {n_rand:,} random controls" if n_rand else ""
    fig.suptitle(
        f"ESM2-8M Attention Around Cleavage Sites by Layer\n"
        f"(averaged over {n_sites:,} real sites{rand_note}, window=±{W})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "attention_profiles.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also save a heatmap version (inbound and outbound stacked)
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    axes2 = axes2.flatten()

    for layer_idx in range(6):
        ax = axes2[layer_idx]
        # Stack inbound and outbound as a 2-row heatmap
        data = np.stack([inbound[layer_idx], outbound[layer_idx]])
        im = ax.imshow(data, aspect="auto", cmap="Reds",
                       extent=[-W - 0.5, W + 0.5, -0.5, 1.5])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Outbound", "Inbound"], fontsize=9)
        ax.set_xlabel("Position relative to cleavage site", fontsize=9)
        ax.set_title(layer_names[layer_idx], fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Attn weight")

    fig2.suptitle(
        f"ESM2-8M Attention Heatmap Around Cleavage Sites\n"
        f"(averaged over {n_sites:,} positions, window=±{W})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, "attention_heatmap.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention analysis around cleavage sites")
    parser.add_argument("--window", type=int, default=15,
                        help="Residues either side of cleavage site (default 15)")
    parser.add_argument("--output", default="plots/attention",
                        help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    sequences = reconstruct_filtered_sequences()
    partitions = pd.read_parquet("processed-data/peptide-partitions.pqt")

    labels_per_protein = []
    for prot_idx in range(len(sequences)):
        prot_rows = partitions[partitions["original-prot"] == float(prot_idx)]
        labels_per_protein.append(prot_rows["y-sites"].to_numpy().astype(int))

    total_sites = sum(l.sum() for l in labels_per_protein)
    print(f"Total proteins: {len(sequences)} | Total cleavage sites: {int(total_sites):,}")

    rng = np.random.default_rng(42)

    print("\nExtracting attention profiles for REAL cleavage sites...")
    inbound, outbound, n_sites = extract_attention_profiles(
        sequences, labels_per_protein, args.window, device, args.batch_size,
        use_random=False, rng=rng,
    )

    print("\nExtracting attention profiles for RANDOM control positions...")
    inbound_rand, outbound_rand, n_rand = extract_attention_profiles(
        sequences, labels_per_protein, args.window, device, args.batch_size,
        use_random=True, rng=rng,
    )

    print(f"\nPlotting (real: {n_sites:,} sites | random: {n_rand:,} positions)...")
    plot_attention_profiles(
        inbound, outbound, args.window, n_sites, args.output,
        inbound_rand=inbound_rand, outbound_rand=outbound_rand, n_rand=n_rand,
    )


if __name__ == "__main__":
    main()
