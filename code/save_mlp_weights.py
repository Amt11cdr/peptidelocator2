"""
Train MLP heads on frozen ESM2-8M embeddings and save weights to models/.

Runs 1 seed × 5-fold CV, saves the fold with the best validation MCC.
Output:
    models/sites_head.pt
    models/peptides_head.pt

These are loaded by the Gradio app (app.py) for inference.

Usage:
    python code/save_mlp_weights.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model


# ── Architecture (must match inference.py) ────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor([alpha, 1.0 - alpha]))
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()


class MLPHead(nn.Module):
    def __init__(self, in_features=320, hidden=320, n_layers=2, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(target: str, save_path: str,
                   epochs: int = 10, batch_size: int = 64,
                   lr: float = 1e-3, alpha: float = 0.1, gamma: float = 3.0):
    print(f"\n{'='*50}")
    print(f"Training MLP head for: {target}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_parquet("processed-data/peptide-partitions.pqt")
    df["y"] = df[f"y-{target}"]
    df["x"] = df["x"].map(lambda x: np.array(x, dtype=np.float32))

    best_mcc   = -1.0
    best_state = None

    for fold in range(5):
        train_df = df[df[f"fold-{fold}"] == "train"].reset_index(drop=True)
        valid_df = df[df[f"fold-{fold}"] == "valid"].reset_index(drop=True)

        def make_loader(d, shuffle=False):
            X = torch.from_numpy(np.stack(d["x"].to_numpy()))
            y = torch.from_numpy(d["y"].to_numpy().astype(int))
            return DataLoader(TensorDataset(X, y),
                              batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(train_df, shuffle=True)
        valid_loader = make_loader(valid_df)

        model     = MLPHead().to(device)
        criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in tqdm(range(epochs), desc=f"Fold {fold}"):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                criterion(model(X_batch), y_batch).backward()
                optimizer.step()

        # Validate
        model.eval()
        preds, probas = [], []
        with torch.no_grad():
            for X_batch, _ in valid_loader:
                logits = model(X_batch.to(device))
                proba  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probas.extend(proba)
                preds.extend((proba > 0.5).astype(int))

        result = evaluate_model(valid_df["y"], preds, probas)
        mcc = result["mcc"]
        print(f"  Fold {fold} valid MCC: {mcc:.3f}")

        if mcc > best_mcc:
            best_mcc   = mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    print(f"\nBest MCC: {best_mcc:.3f} — saved to {save_path}")


if __name__ == "__main__":
    train_and_save("sites",    "models/sites_head.pt")
    train_and_save("peptides", "models/peptides_head.pt")
    print("\nDone. Both heads saved to models/")
