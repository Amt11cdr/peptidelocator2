"""
PCA of per-residue ESM2-8M embeddings at each of the 6 transformer layers.

For each layer (0-5), collects embeddings for a balanced subsample of cleavage
site and non-cleavage residues, runs PCA, and saves a 2x3 grid plot.

Usage:
    python code/interpret_umap.py
    python code/interpret_umap.py --n-samples 3000 --output plots/umap
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

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


# ── Embedding extraction ───────────────────────────────────────────────────────

def get_layer_embeddings(sequences, labels_per_protein, device, batch_size=16):
    """
    Run ESM2-8M forward pass and collect per-residue hidden states at all 6 layers.

    Args:
        sequences: list of amino acid sequence strings
        labels_per_protein: list of np.arrays (one per protein), 1=cleavage site
        device: torch device
        batch_size: number of sequences per forward pass

    Returns:
        layer_embeddings: dict {layer_idx: np.array (n_residues, 320)}
        all_labels: np.array (n_residues,)
    """
    print("Loading ESM2-8M from HuggingFace...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    model.eval()

    # 7 sets of hidden states: embedding layer + 6 transformer layers
    layer_reps = {i: [] for i in range(7)}
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
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
                output_hidden_states=True,
            )

            # hidden_states: tuple of (batch, seq_len+2, 320) — +2 for BOS/EOS tokens
            for b_idx, (seq, labels) in enumerate(zip(batch_seqs, batch_labels)):
                seq_len = len(seq)
                for layer_idx in range(7):
                    # Slice off BOS token (index 0) and EOS token (index seq_len+1)
                    rep = outputs.hidden_states[layer_idx][b_idx, 1:seq_len + 1, :]
                    layer_reps[layer_idx].append(rep.cpu().numpy())
                all_labels.append(labels)

            if i % (batch_size * 10) == 0:
                print(f"  Processed {i}/{len(sequences)} proteins")

    # Stack everything
    layer_embeddings = {}
    for layer_idx in range(7):
        layer_embeddings[layer_idx] = np.vstack(layer_reps[layer_idx])

    all_labels = np.concatenate(all_labels)
    return layer_embeddings, all_labels


# ── Subsampling ───────────────────────────────────────────────────────────────

def balanced_subsample(embeddings_dict, labels, n_per_class=2000, seed=42):
    """Subsample n_per_class residues from each class for each layer."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = min(n_per_class, len(pos_idx))
    n_neg = min(n_per_class, len(neg_idx))

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])

    sub_labels = labels[chosen]
    sub_embeddings = {k: v[chosen] for k, v in embeddings_dict.items()}
    return sub_embeddings, sub_labels, chosen_pos, chosen_neg


# ── UMAP + plotting ───────────────────────────────────────────────────────────

def run_and_plot(layer_embeddings, labels, output_dir, n_per_class):
    os.makedirs(output_dir, exist_ok=True)

    sub_embeddings, sub_labels, _, _ = balanced_subsample(
        layer_embeddings, labels, n_per_class=n_per_class
    )

    layer_names = [
        "Layer 0 — no class structure",
        "Layer 1 — cluster formation begins",
        "Layer 2 — mixed clustering, no separation",
        "Layer 3 — cleavage signal emerging",
        "Layer 4 — peak class separation",
        "Layer 5 — stable separation",
    ]

    colors = {0: "#90CAF9", 1: "#EF9A9A"}  # blue=non-cleavage, red=cleavage
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for layer_idx in range(6):
        print(f"  Running PCA for layer {layer_idx}...")
        ax = axes[layer_idx]
        X = sub_embeddings[layer_idx + 1]  # +1 because index 0 is embedding layer

        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(X)
        var1, var2 = pca.explained_variance_ratio_ * 100

        for label_val, color in colors.items():
            mask = sub_labels == label_val
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=color,
                s=2,
                alpha=0.5,
                rasterized=True,
            )

        ax.set_title(layer_names[layer_idx], fontsize=11, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=9)
        ax.tick_params(labelsize=7)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#90CAF9", label="Non-cleavage residue"),
        mpatches.Patch(color="#EF9A9A", label="Cleavage site residue"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "ESM2-8M Per-Residue Representations by Layer\n"
        f"(PCA, {n_per_class} residues per class)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "pca_layers.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCA of ESM2 layer embeddings")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Residues per class for UMAP (default 2000)")
    parser.add_argument("--output", default="plots/umap",
                        help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    sequences = reconstruct_filtered_sequences()
    partitions = pd.read_parquet("processed-data/peptide-partitions.pqt")

    # Group labels by protein (original-prot index)
    labels_per_protein = []
    for prot_idx in range(len(sequences)):
        prot_rows = partitions[partitions["original-prot"] == float(prot_idx)]
        labels_per_protein.append(prot_rows["y-sites"].to_numpy().astype(int))

    print(f"Total proteins: {len(sequences)}")
    total_residues = sum(len(l) for l in labels_per_protein)
    total_sites = sum(l.sum() for l in labels_per_protein)
    print(f"Total residues: {total_residues:,} | Cleavage sites: {int(total_sites):,}")

    print("\nExtracting layer embeddings (this may take a while)...")
    layer_embeddings, all_labels = get_layer_embeddings(
        sequences, labels_per_protein, device, batch_size=args.batch_size
    )

    print("\nRunning UMAP and plotting...")
    run_and_plot(layer_embeddings, all_labels, args.output, args.n_samples)


if __name__ == "__main__":
    main()
