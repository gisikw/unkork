"""Tests for voice tensor operations."""

import numpy as np
import torch

from unkork.tensors import VOICE_FLAT_DIM, VOICE_SHAPE, blend, flatten, random_blends, unflatten


def _make_voice() -> torch.Tensor:
    return torch.randn(VOICE_SHAPE)


def test_flatten_shape():
    t = _make_voice()
    flat = flatten(t)
    assert flat.shape == (VOICE_FLAT_DIM,)
    assert flat.dtype == np.float32


def test_unflatten_shape():
    flat = np.random.randn(VOICE_FLAT_DIM).astype(np.float32)
    t = unflatten(flat)
    assert t.shape == VOICE_SHAPE


def test_flatten_unflatten_roundtrip():
    original = _make_voice()
    reconstructed = unflatten(flatten(original))
    assert torch.allclose(original, reconstructed, atol=1e-6)


def test_blend_two_voices():
    v1 = torch.ones(VOICE_SHAPE) * 2.0
    v2 = torch.ones(VOICE_SHAPE) * 4.0
    weights = np.array([0.5, 0.5])
    result = blend([v1, v2], weights)
    assert result.shape == VOICE_SHAPE
    assert torch.allclose(result, torch.ones(VOICE_SHAPE) * 3.0, atol=1e-6)


def test_blend_normalizes_weights():
    v1 = torch.zeros(VOICE_SHAPE)
    v2 = torch.ones(VOICE_SHAPE) * 10.0
    # Weights 2:8 should normalize to 0.2:0.8 -> result = 8.0
    weights = np.array([2.0, 8.0])
    result = blend([v1, v2], weights)
    expected = torch.ones(VOICE_SHAPE) * 8.0
    assert torch.allclose(result, expected, atol=1e-6)


def test_random_blends_count():
    sources = [_make_voice() for _ in range(10)]
    blends = random_blends(sources, count=20, rng=np.random.default_rng(42))
    assert len(blends) == 20
    for b in blends:
        assert b.shape == VOICE_SHAPE


def test_random_blends_deterministic():
    sources = [_make_voice() for _ in range(10)]
    a = random_blends(sources, count=5, rng=np.random.default_rng(42))
    b = random_blends(sources, count=5, rng=np.random.default_rng(42))
    for ta, tb in zip(a, b):
        assert torch.allclose(ta, tb)


def test_load_save_roundtrip(tmp_path):
    from unkork.tensors import load_voice, save_voice

    original = _make_voice()
    path = tmp_path / "test_voice.pt"
    save_voice(original, path)
    loaded = load_voice(path)
    assert torch.allclose(original, loaded)
