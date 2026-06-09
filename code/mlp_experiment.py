import os
import yaml

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import typer

from evaluate import evaluate_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch, num_classes)
        # targets: (batch,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MLP(nn.Module):
    def __init__(self, in_features, hidden_state, layers, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_state)  # adjust based on input length
        self.layers = nn.ModuleList([nn.Linear(hidden_state, hidden_state)
                                             for n in range(layers)])
        self.final = nn.Linear(hidden_state, num_classes)
        self.final_af = nn.Softmax(-1)

    def forward(self, x):
        # x: (batch, channels, length)
        x = F.relu(self.fc1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.final_af(self.final(x))

        return x


def run_experiment(
    target: str,
    layers: int = 2,
    alpha_weight: float = 0.1,
    gamma: float = 3.0
):
    df = pd.read_parquet("processed-data/peptide-partitions.pqt")
    df['y'] = df[f'y-{target}']
    os.makedirs('results', exist_ok=True)
    df['x'] = df['x'].map(lambda x: np.array(x, dtype=np.float32))

    all_numbers = []
    for n in os.listdir('results'):
        if not n.endswith('.csv') or target not in n:
            continue
        exp_num = int(n.split('-')[-1].replace(".csv", ""))
        all_numbers.append(exp_num)
    all_numbers.append(0)
    curr_number = max(all_numbers)

    metadata = {
        "target": target,
        "model": "nn",
        "hpo": "no",
        "downsampling": "no",
        "layers": layers,
        "hidden_state": 320,
        "alpha": [alpha_weight, 1 - alpha_weight],
        "gamma": gamma,
        "exp-number": curr_number + 1
    }

    all_results = []
    for seed in range(5):
        for fold in range(5):
            print(f"Running Fold {fold} with seed {seed}")
            train_df = df[df[f'fold-{fold}'] == 'train']
            valid_df = df[df[f'fold-{fold}'] == 'valid']
            test_df = df[df[f'fold-{fold}'] == 'test']

            train_df.reset_index(inplace=True, drop=True)
            valid_df.reset_index(inplace=True, drop=True)
            test_df.reset_index(inplace=True, drop=True)

            dataset = TensorDataset(
                torch.from_numpy(np.stack(train_df['x'].to_numpy()).astype(np.float32)),
                torch.from_numpy(train_df['y'].to_numpy().astype(int)))
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(np.stack(test_df['x'].to_numpy()).astype(np.float32)),
                    torch.from_numpy(test_df['y'].to_numpy().astype(int))),
                    batch_size=64, shuffle=False
                )
            valid_df_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(np.stack(valid_df['x'].to_numpy()).astype(np.float32)),
                    torch.from_numpy(valid_df['y'].to_numpy().astype(int))),
                    batch_size=64, shuffle=False
                )
            # Model, loss, optimizer
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")

            model = MLP(in_features=320, hidden_state=metadata['hidden_state'], layers=metadata['layers'], num_classes=2).to(device)
            criterion = FocalLoss(alpha=torch.tensor(metadata['alpha']).to(device), gamma=metadata['gamma'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training
            epochs = 10
            pbar = tqdm(range(epochs))

            for epoch in pbar:
                model.train()
                total_loss = 0

                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f'Loss: {loss:.2f}')

                    total_loss += loss.item()
                all_preds = []
                model.eval()
                for inputs, targets in valid_df_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = (model(inputs)).detach().cpu().numpy()
                    all_preds.append(preds)

                result = evaluate_model(
                    y_true=valid_df['y'],
                    y_pred=(np.concatenate(all_preds) > 0.5)[:, 1],
                    y_pred_proba=np.concatenate(all_preds)[:, 1]
                )
                result = {f'valid_{k}': v for k, v in result.items()}
                result['fold'] = fold
                result['seed'] = seed
                result['epoch'] = epoch

                print(f"MCC: {result['valid_mcc']:.2f} - Recall: {result['valid_recall_weighted']:.2f}")
                print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
                all_results.append(result)

            all_preds = []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = (model(inputs)).detach().cpu().numpy()
                all_preds.append(preds)

            result = evaluate_model(
                y_true=test_df['y'],
                y_pred=(np.concatenate(all_preds) > 0.5)[:, 1],
                y_pred_proba=np.concatenate(all_preds)[:, 1]
            )
            result = {f'{k}': v for k, v in result.items()}

            print(f"MCC: {result['mcc']:.2f} - Recall: {result['recall_weighted']:.2f}")
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
            all_results.append(result)

            # rf.fit(np.stack(train_df['x']), train_df['y'])
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(f"results/results-{target}-predictor-{metadata['exp-number']}.csv", index=False)
    yaml.safe_dump(metadata, open(f"results/{target}-{metadata['exp-number']}.yml", 'w'))
    print(f"MCC: {result_df['mcc'].mean():.2f}±{result_df['mcc'].std():.2f}")
    print(f"Weighted precision: {result_df['precision_weighted'].mean():.2f}±{result_df['precision_weighted'].std():.2f}")
    print(f"Recall: {result_df['recall_macro'].mean():.2f}±{result_df['recall_macro'].std():.2f}")


if __name__ == "__main__":
    typer.run(run_experiment)
