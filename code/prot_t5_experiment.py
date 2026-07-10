"""
ProtT5-XL / ProstT5 embedding experiment.

Extracts per-residue embeddings from a T5-based protein language model,
caches them to parquet, then runs the same frozen-embedding MLP experiment
used in esm2_size_experiment.py.

Supported models:
  prot_t5    →  Rostlab/prot_t5_xl_uniref50   (3B params, 1024-dim)
  prost_t5   →  Rostlab/ProstT5               (3B params, 1024-dim, structural fine-tune)

Usage:
    python code/prot_t5_experiment.py sites --model prot_t5
    python code/prot_t5_experiment.py peptides --model prost_t5
"""

import os
import re
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import typer

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_IDS = {
    "prot_t5":  "Rostlab/prot_t5_xl_uniref50",
    "prost_t5": "Rostlab/ProstT5",
}
EMBEDDING_DIM = 1024  # both models output 1024-dim per residue

PARTITIONS_BASE = "processed-data/peptide-partitions.pqt"


# ── MLP (same architecture as size experiment) ────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor([alpha, 1.0 - alpha]))
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()


class MLP(nn.Module):
    def __init__(self, in_features=1024, hidden=1024, n_layers=2, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.hidden = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.softmax(self.out(x))


# ── Data loading ──────────────────────────────────────────────────────────────

def _reconstruct_sequences():
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
                end   = int(site.split("-")[1])
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


# ── T5 embedding computation ──────────────────────────────────────────────────

def compute_and_cache_embeddings(model_key: str) -> str:
    """Compute per-residue T5 embeddings and cache to parquet.

    T5 tokenization notes:
      - Amino acids must have a space between each character.
      - Rare AAs (U, Z, O, B) are replaced with X.
      - The encoder output has seq_len + 1 tokens (one EOS appended).
        We take positions 0:seq_len.

    Returns:
        path to the cached partitions parquet file
    """
    from transformers import T5Tokenizer, T5EncoderModel

    cache_path = f"processed-data/peptide-partitions-{model_key}.pqt"
    if os.path.exists(cache_path):
        print(f"Using cached embeddings: {cache_path}")
        return cache_path

    hf_id = MODEL_IDS[model_key]
    print(f"Loading {model_key} ({hf_id})...")

    tokenizer = T5Tokenizer.from_pretrained(hf_id, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(hf_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load in fp16 to save VRAM — ProtT5-XL is 3B params (~12 GB fp32, ~6 GB fp16)
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    sequences = _reconstruct_sequences()
    print(f"  {len(sequences)} proteins to embed")

    # Process one protein at a time to avoid OOM with 3B-param model
    all_reps = []
    with torch.no_grad():
        for i, seq in enumerate(tqdm(sequences, desc="Embedding")):
            # Replace rare AAs; add space between each residue for T5 tokenizer
            seq_clean = re.sub(r"[UZOB]", "X", seq)
            seq_spaced = " ".join(list(seq_clean))

            ids = tokenizer(
                seq_spaced,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            output = model(**ids)
            # last_hidden_state: (1, seq_len+1, 1024) — +1 for EOS token
            rep = output.last_hidden_state[0, :len(seq), :].cpu().float().numpy()
            all_reps.append(rep)

    # Flatten to per-residue rows
    all_reps_flat = np.vstack(all_reps).astype(np.float32)
    print(f"  Embedding shape: {all_reps_flat.shape}")

    partitions = pd.read_parquet(PARTITIONS_BASE)
    assert len(partitions) == len(all_reps_flat), (
        f"Mismatch: partitions={len(partitions)}, embeddings={len(all_reps_flat)}"
    )
    partitions["x"] = [row.tolist() for row in all_reps_flat]
    os.makedirs("processed-data", exist_ok=True)
    partitions.to_parquet(cache_path)
    print(f"  Saved: {cache_path}")
    return cache_path


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    target: str,
    model:  str   = typer.Option("prot_t5", help="prot_t5 or prost_t5"),
    layers: int   = typer.Option(2,          help="MLP hidden layers"),
    epochs: int   = typer.Option(10,         help="Training epochs"),
    batch_size: int = typer.Option(64,       help="Batch size"),
    lr:     float = typer.Option(1e-3,       help="Learning rate"),
    alpha_weight: float = typer.Option(0.1,  help="Focal loss alpha"),
    gamma:  float = typer.Option(3.0,        help="Focal loss gamma"),
):
    """Run frozen ProtT5 / ProstT5 embedding experiment."""
    assert target in ("sites", "peptides"), "target must be 'sites' or 'peptides'"
    assert model in MODEL_IDS, f"--model must be one of {list(MODEL_IDS)}"

    # Get or compute embeddings
    data_path = compute_and_cache_embeddings(model)

    df = pd.read_parquet(data_path)
    df["y"] = df[f"y-{target}"]
    df["x"] = df["x"].map(lambda x: np.array(x, dtype=np.float32))

    os.makedirs("results", exist_ok=True)
    existing = [
        int(f.split("-")[-1].replace(".csv", ""))
        for f in os.listdir("results")
        if f.endswith(".csv") and target in f
    ]
    exp_num = max(existing, default=0) + 1

    metadata = {
        "target":      target,
        "model":       model,
        "hf_id":       MODEL_IDS[model],
        "loss_type":   "focal",
        "downsampling": "no",
        "layers":      layers,
        "hidden_state": EMBEDDING_DIM,
        "alpha":       [alpha_weight, 1 - alpha_weight],
        "gamma":       gamma,
        "epochs":      epochs,
        "batch_size":  batch_size,
        "lr":          lr,
        "exp-number":  exp_num,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device} | {model} | target={target} | exp={exp_num}")

    all_results = []
    for seed in range(5):
        for fold in range(5):
            print(f"\nFold {fold} | Seed {seed}")

            train_df = df[df[f"fold-{fold}"] == "train"].reset_index(drop=True)
            valid_df = df[df[f"fold-{fold}"] == "valid"].reset_index(drop=True)
            test_df  = df[df[f"fold-{fold}"] == "test"].reset_index(drop=True)

            def make_loader(d, shuffle=False):
                X = torch.from_numpy(np.stack(d["x"].to_numpy()).astype(np.float32))
                y = torch.from_numpy(d["y"].to_numpy().astype(int))
                return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                                  shuffle=shuffle)

            train_loader = make_loader(train_df, shuffle=True)
            valid_loader = make_loader(valid_df)
            test_loader  = make_loader(test_df)

            model_nn = MLP(in_features=EMBEDDING_DIM, hidden=EMBEDDING_DIM,
                           n_layers=layers).to(device)
            criterion = FocalLoss(alpha=alpha_weight, gamma=gamma).to(device)
            optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)

            for epoch in tqdm(range(epochs), desc=f"F{fold}S{seed}"):
                model_nn.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    criterion(model_nn(X_batch), y_batch).backward()
                    optimizer.step()

                model_nn.eval()
                preds = []
                with torch.no_grad():
                    for X_batch, _ in valid_loader:
                        preds.append(model_nn(X_batch.to(device)).cpu().numpy())
                preds = np.concatenate(preds)
                result = evaluate_model(valid_df["y"], (preds > 0.5)[:, 1], preds[:, 1])
                result = {f"valid_{k}": v for k, v in result.items()}
                result.update({"fold": fold, "seed": seed, "epoch": epoch})
                all_results.append(result)
                print(f"  Epoch {epoch+1} | Valid MCC: {result['valid_mcc']:.3f}")

            # Test
            model_nn.eval()
            preds = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    preds.append(model_nn(X_batch.to(device)).cpu().numpy())
            preds = np.concatenate(preds)
            result = evaluate_model(test_df["y"], (preds > 0.5)[:, 1], preds[:, 1])
            result.update({"fold": fold, "seed": seed})
            all_results.append(result)
            print(f"  Test MCC: {result['mcc']:.3f}")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"results/results-{target}-predictor-{exp_num}.csv", index=False)
    yaml.safe_dump(metadata, open(f"results/{target}-{exp_num}.yml", "w"))

    test_rows = result_df[result_df["mcc"].notna()]
    print(f"\n=== {model} | {target} ===")
    print(f"MCC:  {test_rows['mcc'].mean():.3f} ± {test_rows['mcc'].std():.3f}")
    print(f"Prec: {test_rows['precision_weighted'].mean():.3f} ± {test_rows['precision_weighted'].std():.3f}")
    print(f"Rec:  {test_rows['recall_macro'].mean():.3f} ± {test_rows['recall_macro'].std():.3f}")


if __name__ == "__main__":
    typer.run(run_experiment)
