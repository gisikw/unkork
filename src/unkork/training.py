"""Training loop for the voice regression codec."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unkork.model import VoiceCodec


@dataclass
class TrainResult:
    """Training outcome."""

    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    best_epoch: int


def build_datasets(
    embeddings: np.ndarray,
    targets: np.ndarray,
    val_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    """Split embeddings/targets into train and validation TensorDatasets.

    Args:
        embeddings: (n_samples, 256) speaker embeddings.
        targets: (n_samples, n_components) PCA-projected voice tensors.
        val_fraction: Fraction of data for validation.

    Returns:
        (train_dataset, val_dataset)
    """
    rng = rng or np.random.default_rng()
    n = len(embeddings)
    n_val = max(1, int(n * val_fraction))

    indices = rng.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def make_ds(idx: np.ndarray) -> TensorDataset:
        x = torch.tensor(embeddings[idx], dtype=torch.float32)
        y = torch.tensor(targets[idx], dtype=torch.float32)
        return TensorDataset(x, y)

    return make_ds(train_idx), make_ds(val_idx)


def train_codec(
    model: VoiceCodec,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainResult:
    """Train the codec model.

    Returns TrainResult with loss history and best model state (loaded in-place).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = model.state_dict()

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best model
    model.load_state_dict(best_state)

    return TrainResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )
