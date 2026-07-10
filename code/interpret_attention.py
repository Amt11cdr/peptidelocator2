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

def extract_attention_profiles(sequences, labels_per_protein, window, device, batch_size=8):
    """
    For each cleavage site, extract a centred attention window at each layer.

    Aggregates two signals:
      - inbound:  how much other residues attend TO the cleavage site
      - outbound: how much the cleavage site attends TO other residues

    Returns:
        inbound_profiles:  {layer: np.array (2*window+1,)}
        outbound_profiles: {layer: np.array (2*window+1,)}
        n_sites: int
    """
    print("Loading ESM2-8M from HuggingFace...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained(
        "facebook/esm2_t6_8M_UR50D",
        attn_implementation="eager"
    ).to(device)
    model.eval()

    W = window
    size = 2 * W + 1
    # Sum over all cleavage sites; divide at the end
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

            outputs = model(
                **tokens,
                output_attentions=True,
            )

            # attentions: tuple of 6 tensors, each (batch, num_heads, seq+2, seq+2)
            for b_idx, (seq, labels) in enumerate(zip(batch_seqs, batch_labels)):
                seq_len = len(seq)
                site_positions = np.where(labels == 1)[0]

                for layer_idx in range(6):
                    # attn shape: (num_heads, seq+2, seq+2) — +2 for BOS/EOS
                    attn = outputs.attentions[layer_idx][b_idx]
                    # Average over heads, strip BOS/EOS → (seq_len, seq_len)
                    attn_mean = attn[:, 1:seq_len + 1, 1:seq_len + 1].mean(dim=0).cpu().numpy()

                    for site_pos in site_positions:
                        # Window bounds (clamped to sequence)
                        start = site_pos - W
                        end   = site_pos + W + 1

                        # Relative indices in the output array
                        out_start = max(0, W - site_pos)
                        out_end   = out_start + min(end, seq_len) - max(start, 0)

                        # Actual sequence indices
                        seq_start = max(0, start)
                        seq_end   = min(seq_len, end)

                        # Inbound: column at site_pos (other residues → cleavage site)
                        inbound_sum[layer_idx][out_start:out_end] += \
                            attn_mean[seq_start:seq_end, site_pos]

                        # Outbound: row at site_pos (cleavage site → other residues)
                        outbound_sum[layer_idx][out_start:out_end] += \
                            attn_mean[site_pos, seq_start:seq_end]

                        n_sites += 1 if layer_idx == 0 else 0  # count once

            if i % (batch_size * 10) == 0:
                print(f"  Processed {i}/{len(sequences)} proteins")

    inbound_profiles  = {k: v / n_sites for k, v in inbound_sum.items()}
    outbound_profiles = {k: v / n_sites for k, v in outbound_sum.items()}
    return inbound_profiles, outbound_profiles, n_sites


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_attention_profiles(inbound, outbound, window, n_sites, output_dir):
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
                label="Inbound (others → site)")
        ax.plot(x, outbound[layer_idx], color="#42A5F5", linewidth=2,
                label="Outbound (site → others)")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_title(layer_names[layer_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Position relative to cleavage site", fontsize=9)
        ax.set_ylabel("Mean attention weight", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"ESM2-8M Attention Around Cleavage Sites by Layer\n"
        f"(averaged over {n_sites:,} cleavage site positions, window=±{W})",
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

    print("\nExtracting attention profiles (this may take a while)...")
    inbound, outbound, n_sites = extract_attention_profiles(
        sequences, labels_per_protein, args.window, device, args.batch_size
    )

    print(f"\nPlotting (aggregated over {n_sites:,} cleavage sites)...")
    plot_attention_profiles(inbound, outbound, args.window, n_sites, args.output)


if __name__ == "__main__":
    main()
