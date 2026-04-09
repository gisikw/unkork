"""Tests for scoring pure functions."""

import numpy as np

from unkork.scoring import (
    cosine_similarity,
    harmonic_mean,
    mel_spectrogram,
    mel_spectrogram_distance,
)


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == 1.0


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-7


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == -1.0


def test_cosine_similarity_zero_vector():
    a = np.array([1.0, 2.0])
    b = np.zeros(2)
    assert cosine_similarity(a, b) == 0.0


def test_harmonic_mean_equal_values():
    assert abs(harmonic_mean([2.0, 2.0, 2.0]) - 2.0) < 1e-7


def test_harmonic_mean_penalizes_low():
    hm = harmonic_mean([0.1, 0.9])
    assert hm < 0.5


def test_harmonic_mean_zero_returns_zero():
    assert harmonic_mean([0.0, 1.0]) == 0.0


def test_harmonic_mean_empty_returns_zero():
    assert harmonic_mean([]) == 0.0


def test_mel_spectrogram_shape():
    """Mel-spectrogram of a sine wave has correct shape."""
    sr = 24000
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    mel = mel_spectrogram(audio, sr=sr, n_mels=80)
    assert mel.shape[0] == 80
    assert mel.shape[1] > 0


def test_mel_spectrogram_distance_identical():
    """Identical signals have zero distance."""
    sr = 24000
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    assert mel_spectrogram_distance(audio, audio, sr=sr) == 0.0


def test_mel_spectrogram_distance_different():
    """Different signals have positive distance."""
    sr = 24000
    t = np.arange(sr) / sr
    a = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    assert mel_spectrogram_distance(a, b, sr=sr) > 0.0


def test_mel_spectrogram_distance_different_lengths():
    """Handles signals of different lengths by truncating."""
    sr = 24000
    a = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    b = np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr).astype(np.float32)
    # Should not crash; distance is nonzero due to STFT edge effects
    # but much less than for different frequencies
    dist = mel_spectrogram_distance(a, b, sr=sr)
    different_freq = np.sin(2 * np.pi * 880 * np.arange(sr // 2) / sr).astype(np.float32)
    dist_different = mel_spectrogram_distance(a, different_freq, sr=sr)
    assert dist < dist_different
