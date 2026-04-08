"""Tests for scoring pure functions (no model weights needed)."""

import numpy as np

from unkork.scoring import cosine_similarity, harmonic_mean


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
    # Harmonic mean is dominated by the lowest value
    hm = harmonic_mean([0.1, 0.9])
    assert hm < 0.5  # arithmetic mean would be 0.5


def test_harmonic_mean_zero_returns_zero():
    assert harmonic_mean([0.0, 1.0]) == 0.0


def test_harmonic_mean_empty_returns_zero():
    assert harmonic_mean([]) == 0.0
