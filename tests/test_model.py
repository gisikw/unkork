"""Tests for MLP model architecture."""

import torch

from unkork.model import VoiceCodec, load_checkpoint, save_checkpoint


def test_forward_shape():
    model = VoiceCodec(input_dim=256, hidden_dim=512, output_dim=50)
    x = torch.randn(8, 256)  # batch of 8
    y = model(x)
    assert y.shape == (8, 50)


def test_forward_single():
    model = VoiceCodec(input_dim=256, hidden_dim=512, output_dim=50)
    x = torch.randn(1, 256)
    y = model(x)
    assert y.shape == (1, 50)


def test_custom_dims():
    model = VoiceCodec(input_dim=128, hidden_dim=256, output_dim=20)
    x = torch.randn(4, 128)
    y = model(x)
    assert y.shape == (4, 20)


def test_parameter_count():
    model = VoiceCodec(input_dim=256, hidden_dim=512, output_dim=50)
    n_params = sum(p.numel() for p in model.parameters())
    # 256*512 + 512 + 512*512 + 512 + 512*50 + 50
    expected = 256 * 512 + 512 + 512 * 512 + 512 + 512 * 50 + 50
    assert n_params == expected


def test_save_load_roundtrip(tmp_path):
    model = VoiceCodec(input_dim=256, hidden_dim=512, output_dim=50)
    path = str(tmp_path / "codec.pt")

    save_checkpoint(model, path, n_components=50, best_epoch=42)
    loaded, metadata = load_checkpoint(path)

    assert metadata["n_components"] == 50
    assert metadata["best_epoch"] == 42

    # Weights should match
    x = torch.randn(4, 256)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
