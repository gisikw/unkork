"""Tests for the training loop with synthetic data."""

import numpy as np

from unkork.model import VoiceCodec
from unkork.training import build_datasets, train_codec


def _make_synthetic_data(
    n_samples: int = 200,
    input_dim: int = 256,
    output_dim: int = 50,
):
    """Generate synthetic regression data: y = Wx + noise."""
    rng = np.random.default_rng(42)
    W = rng.standard_normal((input_dim, output_dim)).astype(np.float32) * 0.1
    x = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    noise = rng.standard_normal((n_samples, output_dim)).astype(np.float32) * 0.01
    y = x @ W + noise
    return x, y


def test_build_datasets_split():
    x, y = _make_synthetic_data(100)
    train_ds, val_ds = build_datasets(x, y, val_fraction=0.2, rng=np.random.default_rng(42))
    assert len(train_ds) == 80
    assert len(val_ds) == 20


def test_training_reduces_loss():
    """Training on synthetic data should reduce validation loss."""
    x, y = _make_synthetic_data(200)
    train_ds, val_ds = build_datasets(x, y, val_fraction=0.1, rng=np.random.default_rng(42))

    model = VoiceCodec(input_dim=256, hidden_dim=128, output_dim=50)
    result = train_codec(model, train_ds, val_ds, epochs=50, batch_size=32, lr=1e-3)

    # Loss should decrease
    assert result.val_losses[-1] < result.val_losses[0]
    assert result.best_val_loss < result.val_losses[0]
    assert result.best_epoch >= 0


def test_training_best_model_restored():
    """After training, model should have best weights loaded."""
    x, y = _make_synthetic_data(200)
    train_ds, val_ds = build_datasets(x, y, val_fraction=0.1, rng=np.random.default_rng(42))

    model = VoiceCodec(input_dim=256, hidden_dim=128, output_dim=50)
    result = train_codec(model, train_ds, val_ds, epochs=30, batch_size=32, lr=1e-3)

    # best_epoch should be a valid epoch index
    assert 0 <= result.best_epoch < 30
