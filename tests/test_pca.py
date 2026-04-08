"""Tests for PCA transform operations."""

import numpy as np

from unkork.pca import PCATransform, fit, inverse_transform, load, save, transform


def _make_data(n_samples: int = 100, n_features: int = 50) -> np.ndarray:
    """Generate synthetic data with some structure for PCA to find."""
    rng = np.random.default_rng(42)
    # Low-rank structure: data lives in a ~10-dim subspace
    basis = rng.standard_normal((10, n_features))
    coeffs = rng.standard_normal((n_samples, 10))
    return (coeffs @ basis).astype(np.float32)


def test_fit_shapes():
    data = _make_data(100, 50)
    pca = fit(data, n_components=5)
    assert pca.components.shape == (5, 50)
    assert pca.mean.shape == (50,)
    assert pca.explained_variance_ratio.shape == (5,)


def test_explained_variance_sums_to_less_than_one():
    data = _make_data(100, 50)
    pca = fit(data, n_components=5)
    assert 0 < pca.explained_variance_ratio.sum() <= 1.0


def test_transform_shape():
    data = _make_data(100, 50)
    pca = fit(data, n_components=5)
    projected = transform(pca, data)
    assert projected.shape == (100, 5)


def test_transform_single_sample():
    data = _make_data(100, 50)
    pca = fit(data, n_components=5)
    projected = transform(pca, data[0])
    assert projected.shape == (5,)


def test_roundtrip_reconstruction():
    """PCA roundtrip on low-rank data should reconstruct well."""
    data = _make_data(100, 50)
    pca = fit(data, n_components=10)  # data is rank ~10

    projected = transform(pca, data)
    reconstructed = inverse_transform(pca, projected)

    # Should be close since n_components >= intrinsic rank
    error = np.mean((data - reconstructed) ** 2)
    assert error < 0.1, f"Reconstruction MSE too high: {error}"


def test_save_load_roundtrip(tmp_path):
    data = _make_data(100, 50)
    pca = fit(data, n_components=5)

    path = tmp_path / "pca.npz"
    save(pca, path)
    loaded = load(path)

    np.testing.assert_array_almost_equal(pca.components, loaded.components)
    np.testing.assert_array_almost_equal(pca.mean, loaded.mean)
    np.testing.assert_array_almost_equal(
        pca.explained_variance_ratio, loaded.explained_variance_ratio
    )
