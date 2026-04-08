"""Dataset construction: collect embeddings and tensors into training arrays."""

from pathlib import Path

import numpy as np


def save_dataset(
    embeddings: np.ndarray,
    flat_tensors: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save training dataset to a directory.

    Writes:
        embeddings.npy — (n_samples, 256) speaker embeddings
        tensors.npy — (n_samples, 130816) flattened voice tensors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "tensors.npy", flat_tensors)


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training dataset from a directory.

    Returns:
        (embeddings, flat_tensors) — both as float32 arrays.
    """
    data_dir = Path(data_dir)
    embeddings = np.load(data_dir / "embeddings.npy").astype(np.float32)
    flat_tensors = np.load(data_dir / "tensors.npy").astype(np.float32)
    return embeddings, flat_tensors
