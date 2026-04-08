"""PCA for voice tensor dimensionality reduction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PCATransform:
    """Fitted PCA transform: stores components, mean, and explained variance."""

    components: np.ndarray  # (n_components, n_features)
    mean: np.ndarray  # (n_features,)
    explained_variance_ratio: np.ndarray  # (n_components,)


def fit(flat_tensors: np.ndarray, n_components: int = 50) -> PCATransform:
    """Fit PCA on flattened voice tensors.

    Args:
        flat_tensors: (n_samples, n_features) array of flattened voice tensors.
        n_components: Number of principal components to keep.

    Returns:
        Fitted PCATransform.
    """
    mean = flat_tensors.mean(axis=0)
    centered = flat_tensors - mean

    # SVD on centered data — more numerically stable than covariance eigdecomp
    u, s, vt = np.linalg.svd(centered, full_matrices=False)

    components = vt[:n_components]
    total_var = (s**2).sum()
    explained_variance_ratio = (s[:n_components] ** 2) / total_var

    return PCATransform(
        components=components,
        mean=mean,
        explained_variance_ratio=explained_variance_ratio,
    )


def transform(pca: PCATransform, flat_tensor: np.ndarray) -> np.ndarray:
    """Project a flattened voice tensor into PCA space.

    Args:
        flat_tensor: (n_features,) or (n_samples, n_features) array.

    Returns:
        (n_components,) or (n_samples, n_components) projected array.
    """
    centered = flat_tensor - pca.mean
    return centered @ pca.components.T


def inverse_transform(pca: PCATransform, projected: np.ndarray) -> np.ndarray:
    """Reconstruct a flattened voice tensor from PCA components.

    Args:
        projected: (n_components,) or (n_samples, n_components) array.

    Returns:
        (n_features,) or (n_samples, n_features) reconstructed array.
    """
    return projected @ pca.components + pca.mean


def save(pca: PCATransform, path: str | Path) -> None:
    """Save PCA transform to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        components=pca.components,
        mean=pca.mean,
        explained_variance_ratio=pca.explained_variance_ratio,
    )


def load(path: str | Path) -> PCATransform:
    """Load PCA transform from a .npz file."""
    data = np.load(path)
    return PCATransform(
        components=data["components"],
        mean=data["mean"],
        explained_variance_ratio=data["explained_variance_ratio"],
    )
