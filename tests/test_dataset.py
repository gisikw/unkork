"""Tests for dataset I/O."""

import numpy as np

from unkork.dataset import load_dataset, save_dataset


def test_save_load_roundtrip(tmp_path):
    embeddings = np.random.randn(50, 256).astype(np.float32)
    tensors = np.random.randn(50, 130816).astype(np.float32)

    save_dataset(embeddings, tensors, tmp_path / "data")
    loaded_emb, loaded_tens = load_dataset(tmp_path / "data")

    np.testing.assert_array_almost_equal(embeddings, loaded_emb)
    np.testing.assert_array_almost_equal(tensors, loaded_tens)


def test_creates_output_dir(tmp_path):
    embeddings = np.random.randn(10, 256).astype(np.float32)
    tensors = np.random.randn(10, 130816).astype(np.float32)

    nested = tmp_path / "a" / "b" / "c"
    save_dataset(embeddings, tensors, nested)
    loaded_emb, loaded_tens = load_dataset(nested)

    assert loaded_emb.shape == (10, 256)
    assert loaded_tens.shape == (10, 130816)
