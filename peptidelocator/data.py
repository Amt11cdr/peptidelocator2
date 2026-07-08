import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_partitions(path: str) -> pd.DataFrame:
    """Load the pre-computed partitions parquet file."""
    df = pd.read_parquet(path)
    df["x"] = df["x"].map(lambda x: np.array(x, dtype=np.float32))
    return df


def get_split(df: pd.DataFrame, fold: int, split: str) -> pd.DataFrame:
    """Return rows for a given fold and split ('train', 'valid', 'test')."""
    return df[df[f"fold-{fold}"] == split].reset_index(drop=True)


def undersample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Undersample majority class to match minority class size."""
    minority_count = int(df["y"].sum())
    return pd.concat([
        df[df["y"] == 0].sample(n=minority_count, random_state=seed),
        df[df["y"] == 1]
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_loader(df: pd.DataFrame, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """Create a DataLoader from a DataFrame with 'x' and 'y' columns."""
    X = torch.from_numpy(np.stack(df["x"].to_numpy()).astype(np.float32))
    y = torch.from_numpy(df["y"].to_numpy().astype(int))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
