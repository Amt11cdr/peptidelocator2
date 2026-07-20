"""
ESM2 model size comparison experiment.

For each model size (8m, 150m, 650m):
  1. Checks if a cached partitions file with the right embeddings exists.
  2. If not, recomputes per-residue embeddings from raw sequences and caches them.
  3. Runs the MLP experiment (focal loss, no downsampling) with the correct input dim.

Usage:
    python code/esm2_size_experiment.py <target> --model-size 150m
    python code/esm2_size_experiment.py sites --model-size 650m
"""

import os
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

# Add code dir to path for local evaluate import
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model

# Model name → embedding dimension
# ESM2 names are passed as short sizes; ProtT5 names are passed in full.
MODEL_DIMS = {
    "8m":        320,
    "150m":      640,
    "650m":      1280,
    "prot-t5-xl":  1024,
    "prost-t5":    1024,
}

# ESM2 sizes get prefixed with "esm2-"; ProtT5 names are used as-is.
ESM2_SIZES = {"8m", "150m", "650m"}

PARTITIONS_BASE = "processed-data/peptide-partitions.pqt"
_CACHE_DIR = os.environ.get("PEPTIDE_CACHE_DIR", "processed-data")
CACHED_PARTITIONS = _CACHE_DIR + "/peptide-partitions-{model_size}.pqt"


# ── Model ────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MLP(nn.Module):
    def __init__(self, in_features, hidden_state, layers, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_state)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_state, hidden_state) for _ in range(layers)]
        )
        self.final = nn.Linear(hidden_state, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.softmax(self.final(x))


# ── Embedding computation ─────────────────────────────────────────────────────

def _reconstruct_filtered_sequences():
    """Reconstruct the filtered protein sequences in the same order used
    when peptide-partitions.pqt was created (i.e. same as original-prot index)."""
    df = pd.read_csv("processed-data/peptide.tsv", sep="\t")

    # Re-apply the same filtering as prepare-data.py
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


def compute_and_cache_embeddings(model_size: str) -> str:
    """Compute per-residue embeddings using the given ESM2 model size,
    replace the x column in the partitions file, and save the result.

    Returns the path to the cached partitions file.
    """
    from autopeptideml.reps.lms import RepEngineLM

    cache_path = CACHED_PARTITIONS.format(model_size=model_size)
    if os.path.exists(cache_path):
        print(f"Using cached embeddings: {cache_path}")
        return cache_path

    print(f"Computing ESM2-{model_size} embeddings (this may take a while)...")
    sequences = _reconstruct_filtered_sequences()
    print(f"  {len(sequences)} proteins to embed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"esm2-{model_size}" if model_size in ESM2_SIZES else model_size
    re = RepEngineLM(model_name, average_pooling=False)
    re.move_to_device(device)

    # Use smaller batches for larger models
    batch_size = {"8m": 64, "150m": 32, "650m": 8}.get(model_size, 8)
    reps = re.compute_reps(sequences, batch_size=batch_size)

    # Load existing partitions (for fold assignments and labels)
    partitions = pd.read_parquet(PARTITIONS_BASE)

    # Flatten to per-residue, trimming to actual sequence length to drop
    # any special tokens that T5-style models may append.
    all_reps = []
    for seq, rep in zip(sequences, reps):
        seq_len = len(seq)
        rep_arr = np.array(rep)  # shape: (tokens, dim) — may include special tokens
        rep_arr = rep_arr[:seq_len]  # keep only the first seq_len rows
        for residue_rep in rep_arr:
            all_reps.append(residue_rep)
    new_x = np.stack(all_reps).astype(np.float32)
    print(f"  Embedding shape (trimmed): {new_x.shape}")

    if len(partitions) != len(new_x):
        raise ValueError(
            f"Mismatch after trimming: partitions has {len(partitions)} rows but "
            f"embeddings have {len(new_x)} rows. "
            f"Check that _reconstruct_filtered_sequences() returns exactly the "
            f"same proteins as peptide-partitions.pqt."
        )

    partitions["x"] = [row.tolist() for row in new_x]
    os.makedirs("processed-data", exist_ok=True)
    partitions.to_parquet(cache_path)
    print(f"  Saved: {cache_path}")
    return cache_path


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    target: str,
    model_size: str = "8m",
    loss: str = "focal",
    layers: int = 2,
    alpha_weight: float = 0.1,
    gamma: float = 3.0,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    assert model_size in MODEL_DIMS, f"--model-size must be one of {list(MODEL_DIMS)}"
    assert loss in ("focal", "bce"), "--loss must be 'focal' or 'bce'"
    in_features = MODEL_DIMS[model_size]

    # Get (or compute) the partitions file for this model
    if model_size == "8m":
        data_path = PARTITIONS_BASE
    else:
        data_path = compute_and_cache_embeddings(model_size)

    df = pd.read_parquet(data_path)
    df["y"] = df[f"y-{target}"]
    df["x"] = df["x"].map(lambda x: np.array(x, dtype=np.float32))
    os.makedirs("results", exist_ok=True)

    # Auto-increment experiment number
    existing = [
        int(f.split("-")[-1].replace(".csv", ""))
        for f in os.listdir("results")
        if f.endswith(".csv") and target in f
    ]
    exp_num = max(existing, default=0) + 1

    metadata = {
        "target": target,
        "model": f"esm2-{model_size}" if model_size in ESM2_SIZES else model_size,
        "model_size": model_size,
        "loss_type": loss,
        "downsampling": "no",
        "layers": layers,
        "hidden_state": in_features,
        "alpha": [alpha_weight, 1 - alpha_weight],
        "gamma": gamma,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "exp-number": exp_num,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device} | ESM2-{model_size} | target={target} | exp={exp_num}")

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
                return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

            loader       = make_loader(train_df, shuffle=True)
            valid_loader = make_loader(valid_df)
            test_loader  = make_loader(test_df)

            model = MLP(in_features=in_features, hidden_state=in_features,
                        layers=layers).to(device)
            if loss == "focal":
                criterion = FocalLoss(
                    alpha=torch.tensor([alpha_weight, 1 - alpha_weight]).to(device),
                    gamma=gamma,
                )
            else:  # plain BCE — no class weighting
                criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in tqdm(range(epochs), desc=f"F{fold}S{seed}"):
                model.train()
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    criterion(model(inputs), targets).backward()
                    optimizer.step()

                model.eval()
                preds = []
                with torch.no_grad():
                    for inputs, _ in valid_loader:
                        preds.append(model(inputs.to(device)).cpu().numpy())
                preds = np.concatenate(preds)
                result = evaluate_model(valid_df["y"], (preds > 0.5)[:, 1], preds[:, 1])
                result = {f"valid_{k}": v for k, v in result.items()}
                result.update({"fold": fold, "seed": seed, "epoch": epoch})
                all_results.append(result)
                print(f"  Epoch {epoch+1} | Valid MCC: {result['valid_mcc']:.3f}")

            # Test
            model.eval()
            preds = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    preds.append(model(inputs.to(device)).cpu().numpy())
            preds = np.concatenate(preds)
            result = evaluate_model(test_df["y"], (preds > 0.5)[:, 1], preds[:, 1])
            result.update({"fold": fold, "seed": seed})
            all_results.append(result)
            print(f"  Test MCC: {result['mcc']:.3f}")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"results/results-{target}-predictor-{exp_num}.csv", index=False)
    yaml.safe_dump(metadata, open(f"results/{target}-{exp_num}.yml", "w"))

    test_rows = result_df[result_df["mcc"].notna()]
    print(f"\n=== ESM2-{model_size} | {target} ===")
    print(f"MCC:       {test_rows['mcc'].mean():.3f} ± {test_rows['mcc'].std():.3f}")
    print(f"Precision: {test_rows['precision_weighted'].mean():.3f} ± {test_rows['precision_weighted'].std():.3f}")
    print(f"Recall:    {test_rows['recall_macro'].mean():.3f} ± {test_rows['recall_macro'].std():.3f}")


if __name__ == "__main__":
    typer.run(run_experiment)
