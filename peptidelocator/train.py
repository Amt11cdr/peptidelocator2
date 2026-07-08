import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import MLP, FocalLoss
from .data import load_partitions, get_split, undersample, make_loader
from .evaluate import evaluate_model


def run_experiment(
    target: str,
    data_path: str = "processed-data/peptide-partitions.pqt",
    results_dir: str = "results",
    num_layers: int = 2,
    hidden_size: int = 320,
    alpha_weight: float = 0.1,
    gamma: float = 3.0,
    loss_type: str = "focal",
    downsample: bool = False,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_seeds: int = 5,
    num_folds: int = 5,
):
    """Run a full cross-validation experiment.

    Args:
        target: 'sites' or 'peptides'
        data_path: Path to the partitions parquet file
        results_dir: Directory to save results
        num_layers: Number of MLP hidden layers
        hidden_size: MLP hidden layer size
        alpha_weight: Weight for minority class (used in both focal and BCE)
        gamma: Focal loss gamma parameter
        loss_type: 'focal' or 'bce'
        downsample: Whether to undersample the majority class
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        num_seeds: Number of random seeds
        num_folds: Number of CV folds
    """
    df = load_partitions(data_path)
    df["y"] = df[f"y-{target}"]
    os.makedirs(results_dir, exist_ok=True)

    # Auto-increment experiment number
    existing = [
        int(f.split("-")[-1].replace(".csv", ""))
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and target in f
    ]
    exp_num = max(existing, default=0) + 1

    metadata = {
        "target": target,
        "model": "nn",
        "loss_type": loss_type,
        "downsampling": "yes" if downsample else "no",
        "layers": num_layers,
        "hidden_state": hidden_size,
        "alpha": [alpha_weight, 1 - alpha_weight],
        "gamma": gamma,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "exp-number": exp_num,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for seed in range(num_seeds):
        for fold in range(num_folds):
            print(f"Fold {fold} | Seed {seed}")

            train_df = get_split(df, fold, "train")
            valid_df = get_split(df, fold, "valid")
            test_df = get_split(df, fold, "test")

            if downsample:
                train_df = undersample(train_df, seed)

            loader = make_loader(train_df, batch_size=batch_size, shuffle=True)
            valid_loader = make_loader(valid_df, batch_size=batch_size)
            test_loader = make_loader(test_df, batch_size=batch_size)

            model = MLP(
                in_features=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            ).to(device)

            if loss_type == "focal":
                criterion = FocalLoss(
                    alpha=torch.tensor([alpha_weight, 1 - alpha_weight]).to(device),
                    gamma=gamma,
                )
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=torch.tensor([alpha_weight, 1 - alpha_weight]).to(device)
                )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training
            for epoch in tqdm(range(epochs), desc=f"Fold {fold} Seed {seed}"):
                model.train()
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                all_preds = []
                with torch.no_grad():
                    for inputs, _ in valid_loader:
                        all_preds.append(model(inputs.to(device)).cpu().numpy())
                preds = np.concatenate(all_preds)
                result = evaluate_model(
                    y_true=valid_df["y"],
                    y_pred=(preds > 0.5)[:, 1],
                    y_pred_proba=preds[:, 1],
                )
                result = {f"valid_{k}": v for k, v in result.items()}
                result.update({"fold": fold, "seed": seed, "epoch": epoch})
                all_results.append(result)
                print(f"  Epoch {epoch+1} | Valid MCC: {result['valid_mcc']:.3f}")

            # Test
            model.eval()
            all_preds = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    all_preds.append(model(inputs.to(device)).cpu().numpy())
            preds = np.concatenate(all_preds)
            result = evaluate_model(
                y_true=test_df["y"],
                y_pred=(preds > 0.5)[:, 1],
                y_pred_proba=preds[:, 1],
            )
            result.update({"fold": fold, "seed": seed})
            all_results.append(result)
            print(f"  Test MCC: {result['mcc']:.3f}")

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"{results_dir}/results-{target}-predictor-{exp_num}.csv", index=False)
    yaml.safe_dump(metadata, open(f"{results_dir}/{target}-{exp_num}.yml", "w"))

    test_results = result_df[result_df["mcc"].notna()]
    print(f"\nMCC: {test_results['mcc'].mean():.3f} ± {test_results['mcc'].std():.3f}")
    print(f"Precision: {test_results['precision_weighted'].mean():.3f} ± {test_results['precision_weighted'].std():.3f}")
    print(f"Recall: {test_results['recall_macro'].mean():.3f} ± {test_results['recall_macro'].std():.3f}")

    return result_df
