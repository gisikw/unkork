"""Voice tensor operations: load, save, blend, reshape."""

from pathlib import Path

import numpy as np
import torch

STYLE_DIM = 256


def load_voice(path: str | Path) -> torch.Tensor:
    """Load a Kokoro voice tensor from a .pt file.

    Validates shape is (N, 1, 256) — the time dimension N varies
    between Kokoro versions (510 for v0.19, 511 for v1.0).
    """
    tensor = torch.load(path, weights_only=True)
    if tensor.ndim != 3 or tensor.shape[1] != 1 or tensor.shape[2] != STYLE_DIM:
        raise ValueError(f"Expected shape (N, 1, {STYLE_DIM}), got {tensor.shape}")
    return tensor


def save_voice(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a voice tensor to a .pt file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def flatten(tensor: torch.Tensor) -> np.ndarray:
    """Flatten a (N, 1, 256) voice tensor to a 1D float32 array."""
    return tensor.squeeze(1).reshape(-1).numpy().astype(np.float32)


def unflatten(flat: np.ndarray) -> torch.Tensor:
    """Reshape a flat array back to (N, 1, 256) voice tensor.

    Infers N from the array length: N = len(flat) // 256.
    """
    n = len(flat) // STYLE_DIM
    return torch.tensor(flat.reshape(n, STYLE_DIM)).unsqueeze(1)


def blend(tensors: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    """Linearly blend voice tensors with given weights. Weights are normalized."""
    weights = weights / weights.sum()
    result = torch.zeros_like(tensors[0])
    for tensor, w in zip(tensors, weights):
        result += w * tensor
    return result


def random_blends(
    tensors: list[torch.Tensor],
    count: int,
    min_voices: int = 2,
    max_voices: int = 4,
    rng: np.random.Generator | None = None,
) -> list[torch.Tensor]:
    """Generate random blends from a pool of voice tensors.

    Each blend picks min_voices..max_voices random tensors and assigns
    Dirichlet-distributed weights for smooth coverage of the voice space.
    """
    rng = rng or np.random.default_rng()
    n = len(tensors)
    results = []
    for _ in range(count):
        k = rng.integers(min_voices, max_voices + 1)
        indices = rng.choice(n, size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        selected = [tensors[i] for i in indices]
        results.append(blend(selected, weights))
    return results
