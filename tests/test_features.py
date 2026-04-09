"""Tests for spectral feature extraction."""

import numpy as np
import soundfile as sf

from unkork.features import extract_spectral_features, extract_spectral_features_batch


def _make_sine_wav(tmp_path, freq: float = 440.0, sr: int = 24000, duration: float = 1.0):
    """Create a sine wave wav file and return its path."""
    t = np.arange(int(sr * duration)) / sr
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    path = tmp_path / f"sine_{freq:.0f}.wav"
    sf.write(str(path), audio, sr)
    return str(path)


def test_feature_vector_shape(tmp_path):
    """Feature vector has expected dimensionality."""
    path = _make_sine_wav(tmp_path)
    features = extract_spectral_features(path)
    # 20*2 + 2 + 2 + 2 + 7 + 2 + 2 + 3 + 80*2 = 220
    assert features.shape == (220,)
    assert features.dtype == np.float32


def test_feature_vector_no_nans(tmp_path):
    """Feature vector should not contain NaN values."""
    path = _make_sine_wav(tmp_path)
    features = extract_spectral_features(path)
    assert not np.any(np.isnan(features))


def test_different_frequencies_different_features(tmp_path):
    """Different frequencies should produce different feature vectors."""
    path_low = _make_sine_wav(tmp_path, freq=200.0)
    path_high = _make_sine_wav(tmp_path, freq=800.0)
    feat_low = extract_spectral_features(path_low)
    feat_high = extract_spectral_features(path_high)
    assert not np.allclose(feat_low, feat_high)


def test_batch_averages(tmp_path):
    """Batch extraction should average features across files."""
    path1 = _make_sine_wav(tmp_path, freq=300.0)
    path2 = _make_sine_wav(tmp_path, freq=500.0)
    batch_feat = extract_spectral_features_batch([path1, path2])
    # Should be same shape as single extraction
    single_feat = extract_spectral_features(path1)
    assert batch_feat.shape == single_feat.shape
    assert batch_feat.dtype == np.float32


def test_custom_n_mfcc(tmp_path):
    """Custom n_mfcc changes feature vector size."""
    path = _make_sine_wav(tmp_path)
    feat_13 = extract_spectral_features(path, n_mfcc=13)
    feat_20 = extract_spectral_features(path, n_mfcc=20)
    # Difference should be (20-13)*2 = 14 dims
    assert feat_20.shape[0] - feat_13.shape[0] == 14
