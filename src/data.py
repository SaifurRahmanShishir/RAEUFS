"""Data loading and preprocessing utilities for RAEUFS."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import Dataset


def scale_features(X, method: str = "minmax") -> torch.Tensor:
    """Scale a 2D feature matrix and return a float32 CPU tensor."""
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method must be either 'minmax' or 'robust'")

    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    X_scaled = scaler.fit_transform(X_np)
    return torch.tensor(X_scaled, dtype=torch.float32)


class CustomDataset(Dataset):
    """Dataset used by the RAEUFS Lightning module.

    The unsupervised training path returns ``(X[idx], X[idx])`` to preserve
    compatibility with the original training loop.
    """

    def __init__(self, X, Y: Optional[np.ndarray] = None):
        if isinstance(X, torch.Tensor):
            self.X = X.detach().cpu().float()
        else:
            self.X = torch.as_tensor(X, dtype=torch.float32)

        if Y is None:
            self.Y = None
        elif isinstance(Y, torch.Tensor):
            self.Y = Y.detach().cpu().long()
        else:
            self.Y = torch.as_tensor(Y, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx], self.X[idx]


