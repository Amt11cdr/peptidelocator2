"""
End-to-end fine-tuning of ESM2-8M for cleavage site / peptide region prediction.

All ESM2-8M parameters are unfrozen. Two optimizer param groups are used:
  - ESM2 backbone:  lr = 1e-5  (fine-tune the whole model gently)
  - MLP head:       lr = 1e-3  (train the head faster)

Checkpoints are saved per fold to:
    checkpoints/finetune_esm2_8m_{target}_fold{fold}_seed{seed}/

These can be loaded by interpret_umap.py --model-path to run post-finetune PCA.

Usage:
    python code/finetune_esm2.py sites
    python code/finetune_esm2.py peptides --epochs 5
    python code/finetune_esm2.py sites --lr-esm 1e-5 --lr-head 1e-3
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import typer

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model

from transformers import EsmModel, EsmTokenizer


# ── Architecture ──────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """Lightweight MLP head on top of per-residue ESM2 representations."""
    def __init__(self, in_features: int = 320, hidden: int = 320,
                 n_layers: int = 2, n_classes: int = 2):
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
        return self.out(x)  # logits, shape (n_residues, 2)


class ESM2Classifier(nn.Module):
    """ESM2-8M backbone + MLP head. Both are trained jointly."""
    def __init__(self):
        super().__init__()
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.head = MLPHead(in_features=320, hidden=320, n_layers=2)

    def forward(self, input_ids, attention_mask, seq_lengths):
        """
        Args:
            input_ids, attention_mask: standard HuggingFace token tensors
            seq_lengths: list[int] — actual sequence lengths (without BOS/EOS)

        Returns:
            logits: list of (seq_len, 2) tensors, one per protein in the batch
        """
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L+2, 320)

        # Strip BOS (index 0) and EOS (index seq_len+1) for each protein
        logits_per_protein = []
        for b, seq_len in enumerate(seq_lengths):
            rep = hidden[b, 1:seq_len + 1, :]   # (seq_len, 320)
            logits_per_protein.append(self.head(rep))  # (seq_len, 2)
        return logits_per_protein

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.esm.save_pretrained(os.path.join(path, "esm"))
        torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))

    @classmethod
    def load(cls, path: str, device="cpu"):
        m = cls.__new__(cls)
        super(ESM2Classifier, m).__init__()
        m.esm = EsmModel.from_pretrained(os.path.join(path, "esm")).to(device)
        m.head = MLPHead()
        m.head.load_state_dict(
            torch.load(os.path.join(path, "head.pt"), map_location=device)
        )
        m.head = m.head.to(device)
        return m


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=3.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor([alpha, 1 - alpha]))
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()


# ── Data ──────────────────────────────────────────────────────────────────────

def _reconstruct_sequences():
    """Same filtering as prepare-data.py — matches original-prot indices."""
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


class ProteinDataset(Dataset):
    """One item = one protein (sequence + per-residue label array)."""
    def __init__(self, prot_indices, sequences, labels_per_prot):
        self.prot_indices = prot_indices
        self.sequences = sequences
        self.labels    = labels_per_prot  # list of np.arrays

    def __len__(self):
        return len(self.prot_indices)

    def __getitem__(self, idx):
        prot_idx = self.prot_indices[idx]
        return self.sequences[prot_idx], self.labels[prot_idx]


def make_collate(tokenizer, device):
    def collate_fn(batch):
        seqs, labels = zip(*batch)
        tokens = tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        seq_lengths = [len(s) for s in seqs]
        return tokens, seq_lengths, labels
    return collate_fn


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(
    target: str,
    epochs:    int   = typer.Option(5,    help="Training epochs per fold/seed"),
    batch_size: int  = typer.Option(4,    help="Proteins per batch"),
    lr_esm:    float = typer.Option(1e-7, help="LR for unfrozen ESM2 layer 5"),
    lr_head:   float = typer.Option(1e-3, help="LR for MLP head"),
    accum_steps: int = typer.Option(8,    help="Gradient accumulation steps"),
    alpha_weight: float = typer.Option(0.1, help="Focal loss alpha (minority class)"),
    gamma:     float = typer.Option(3.0,  help="Focal loss gamma"),
):
    """Fine-tune ESM2-8M end-to-end for cleavage site / peptide region prediction."""
    assert target in ("sites", "peptides"), "target must be 'sites' or 'peptides'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    sequences = _reconstruct_sequences()
    partitions = pd.read_parquet("processed-data/peptide-partitions.pqt")
    partitions["y"] = partitions[f"y-{target}"]

    # Build per-protein label arrays indexed by original-prot
    labels_per_prot = []
    for prot_idx in range(len(sequences)):
        rows = partitions[partitions["original-prot"] == float(prot_idx)]
        labels_per_prot.append(rows["y"].to_numpy().astype(int))

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    collate_fn = make_collate(tokenizer, device)

    # Get unique protein indices per fold split
    def get_prot_indices(fold, split):
        rows = partitions[partitions[f"fold-{fold}"] == split]
        return sorted(rows["original-prot"].unique().astype(int).tolist())

    # ── Result storage ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    existing = [
        int(f.split("-")[-1].replace(".csv", ""))
        for f in os.listdir("results")
        if f.endswith(".csv") and target in f
    ]
    exp_num = max(existing, default=0) + 1

    metadata = {
        "target":       target,
        "model":        "esm2-8m-layer5-finetuned",
        "loss_type":    "focal",
        "epochs":       epochs,
        "batch_size":   batch_size,
        "lr_esm":       lr_esm,
        "lr_head":      lr_head,
        "accum_steps":  accum_steps,
        "alpha":        [alpha_weight, 1 - alpha_weight],
        "gamma":        gamma,
        "exp-number":   exp_num,
        "finetuned":    True,
    }

    all_results = []

    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)

        for fold in range(5):
            print(f"\n{'='*50}")
            print(f"Seed {seed} | Fold {fold}")

            train_idx = get_prot_indices(fold, "train")
            valid_idx = get_prot_indices(fold, "valid")
            test_idx  = get_prot_indices(fold, "test")

            train_set = ProteinDataset(train_idx, sequences, labels_per_prot)
            valid_set = ProteinDataset(valid_idx, sequences, labels_per_prot)
            test_set  = ProteinDataset(test_idx,  sequences, labels_per_prot)

            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True,  collate_fn=collate_fn)
            valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn)
            test_loader  = DataLoader(test_set,  batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn)

            # Fresh model + optimizer for each seed/fold
            model = ESM2Classifier().to(device)
            criterion = FocalLoss(alpha=alpha_weight, gamma=gamma).to(device)

            # Freeze all ESM2 params, then unfreeze layer 5 only (Raul: 1e-7 lr)
            for param in model.esm.parameters():
                param.requires_grad = False
            for param in model.esm.encoder.layer[5].parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": model.esm.encoder.layer[5].parameters(),
                 "lr": lr_esm, "weight_decay": 0.01},
                {"params": model.head.parameters(),
                 "lr": lr_head, "weight_decay": 0.0},
            ])

            best_valid_mcc = -1.0
            best_ckpt_path = f"checkpoints/finetune_esm2_8m_{target}_fold{fold}_seed{seed}"

            for epoch in range(epochs):
                # ── Train ──
                model.train()
                optimizer.zero_grad()
                running_loss = 0.0
                step = 0

                for batch_idx, (tokens, seq_lengths, batch_labels) in enumerate(
                    tqdm(train_loader, desc=f"F{fold}S{seed} Ep{epoch+1}")
                ):
                    logits_list = model(
                        tokens["input_ids"],
                        tokens["attention_mask"],
                        seq_lengths,
                    )

                    # Flatten logits + labels across all proteins in batch
                    all_logits = torch.cat(logits_list, dim=0)
                    all_targets = torch.cat([
                        torch.tensor(lbl, dtype=torch.long, device=device)
                        for lbl in batch_labels
                    ])

                    loss = criterion(all_logits, all_targets) / accum_steps
                    loss.backward()
                    running_loss += loss.item() * accum_steps

                    step += 1
                    if step % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                # Flush remaining gradients
                if step % accum_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss = running_loss / max(step, 1)

                # ── Validate ──
                model.eval()
                all_preds, all_proba, all_true = [], [], []
                with torch.no_grad():
                    for tokens, seq_lengths, batch_labels in valid_loader:
                        logits_list = model(
                            tokens["input_ids"],
                            tokens["attention_mask"],
                            seq_lengths,
                        )
                        for logits, lbl in zip(logits_list, batch_labels):
                            proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                            preds = (proba > 0.5).astype(int)
                            all_proba.extend(proba)
                            all_preds.extend(preds)
                            all_true.extend(lbl)

                result = evaluate_model(all_true, all_preds, all_proba)
                valid_mcc = result["mcc"]
                print(f"  Epoch {epoch+1} | loss={avg_loss:.4f} | valid MCC={valid_mcc:.3f}")

                result_row = {f"valid_{k}": v for k, v in result.items()}
                result_row.update({"fold": fold, "seed": seed, "epoch": epoch})
                all_results.append(result_row)

                # Save best checkpoint for this fold/seed
                if valid_mcc > best_valid_mcc:
                    best_valid_mcc = valid_mcc
                    model.save(best_ckpt_path)
                    print(f"  ✓ Saved best checkpoint (MCC={valid_mcc:.3f})")

            # ── Test (best checkpoint) ──
            best_model = ESM2Classifier.load(best_ckpt_path, device=device)
            best_model.eval()
            all_preds, all_proba, all_true = [], [], []
            with torch.no_grad():
                for tokens, seq_lengths, batch_labels in test_loader:
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    logits_list = best_model(
                        tokens["input_ids"],
                        tokens["attention_mask"],
                        seq_lengths,
                    )
                    for logits, lbl in zip(logits_list, batch_labels):
                        proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                        preds = (proba > 0.5).astype(int)
                        all_proba.extend(proba)
                        all_preds.extend(preds)
                        all_true.extend(lbl)

            result = evaluate_model(all_true, all_preds, all_proba)
            result.update({"fold": fold, "seed": seed})
            all_results.append(result)
            print(f"  Test MCC: {result['mcc']:.3f} (best valid: {best_valid_mcc:.3f})")

    # ── Save results ──────────────────────────────────────────────────────────
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"results/results-{target}-predictor-{exp_num}.csv", index=False)
    yaml.safe_dump(metadata, open(f"results/{target}-{exp_num}.yml", "w"))

    test_rows = result_df[result_df["mcc"].notna()]
    print(f"\n{'='*50}")
    print(f"=== ESM2-8M Fine-tuned | {target} ===")
    print(f"MCC:  {test_rows['mcc'].mean():.3f} ± {test_rows['mcc'].std():.3f}")
    print(f"Prec: {test_rows['precision_weighted'].mean():.3f} ± {test_rows['precision_weighted'].std():.3f}")
    print(f"Rec:  {test_rows['recall_macro'].mean():.3f} ± {test_rows['recall_macro'].std():.3f}")


if __name__ == "__main__":
    typer.run(run_experiment)
