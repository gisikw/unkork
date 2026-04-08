"""Voice tensor operations: load, save, blend, reshape."""

from pathlib import Path

import numpy as np
import torch

VOICE_SHAPE = (511, 1, 256)
VOICE_FLAT_DIM = 511 * 256  # 130816


def load_voice(path: str | Path) -> torch.Tensor:
    """Load a Kokoro voice tensor from a .pt file."""
    tensor = torch.load(path, weights_only=True)
    if tensor.shape != VOICE_SHAPE:
        raise ValueError(f"Expected shape {VOICE_SHAPE}, got {tensor.shape}")
    return tensor


def save_voice(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a voice tensor to a .pt file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def flatten(tensor: torch.Tensor) -> np.ndarray:
    """Flatten a (511, 1, 256) voice tensor to (130816,) float32 array."""
    return tensor.squeeze(1).reshape(-1).numpy().astype(np.float32)


def unflatten(flat: np.ndarray) -> torch.Tensor:
    """Reshape a flat array back to (511, 1, 256) voice tensor."""
    return torch.tensor(flat.reshape(511, 256)).unsqueeze(1)


def blend(tensors: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    """Linearly blend voice tensors with given weights. Weights are normalized."""
    weights = weights / weights.sum()
    result = torch.zeros(VOICE_SHAPE)
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
