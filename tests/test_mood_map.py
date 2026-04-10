"""Tests for mood mapping experiment."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from unkork.cli import main
from unkork.mood_map import (
    MOODS,
    MOOD_INSTRUCTIONS,
    SENTENCES,
    ClipRecord,
    build_tts_request,
    compute_silhouette,
    explained_variance_report,
    fit_pca_2d,
    fit_tsne_2d,
    load_manifest,
    mood_instruction,
    save_manifest,
    write_report,
    FeatureAnalysis,
)


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def test_mood_instruction_all_moods():
    """Every mood in MOODS has a non-empty instruction string."""
    for mood in MOODS:
        inst = mood_instruction(mood)
        assert isinstance(inst, str)
        assert len(inst) > 10


def test_mood_instruction_unknown_raises():
    """Unknown mood raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mood"):
        mood_instruction("furious")


def test_build_tts_request_fields():
    """Request body has all required fields."""
    body = build_tts_request("Hello.", "sultry", "ryan")
    assert body["model"] == "qwen3-tts"
    assert body["voice"] == "ryan"
    assert body["input"] == "Hello."
    assert body["response_format"] == "wav"
    assert "sultry" not in body["instructions"].lower() or "smooth" in body["instructions"].lower()
    assert len(body["instructions"]) > 0


def test_build_tts_request_custom_instructions():
    """Custom instruction dict overrides defaults."""
    custom = {"angry": "Yell loudly."}
    body = build_tts_request("Hi.", "angry", "ryan", instructions=custom)
    assert body["instructions"] == "Yell loudly."


def test_sentences_are_nonempty():
    """Sentence list has entries, all non-empty strings."""
    assert len(SENTENCES) >= 15
    for s in SENTENCES:
        assert isinstance(s, str)
        assert len(s) > 10


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    """Save and load manifest preserves all records."""
    records = [
        ClipRecord(path="/data/sultry/ryan_000.wav", mood="sultry", voice="ryan", sentence_idx=0),
        ClipRecord(path="/data/neutral/aiden_005.wav", mood="neutral", voice="aiden", sentence_idx=5),
    ]
    manifest_path = tmp_path / "manifest.json"
    save_manifest(records, manifest_path)
    loaded = load_manifest(manifest_path)
    assert len(loaded) == 2
    assert loaded[0].mood == "sultry"
    assert loaded[0].voice == "ryan"
    assert loaded[1].sentence_idx == 5


def test_manifest_json_format(tmp_path):
    """Manifest is valid JSON with expected structure."""
    records = [ClipRecord(path="/a.wav", mood="neutral", voice="ryan", sentence_idx=0)]
    manifest_path = tmp_path / "manifest.json"
    save_manifest(records, manifest_path)
    data = json.loads(manifest_path.read_text())
    assert isinstance(data, list)
    assert data[0]["mood"] == "neutral"


# ---------------------------------------------------------------------------
# Analysis (pure functions on synthetic data)
# ---------------------------------------------------------------------------


def test_fit_pca_2d_shape():
    """PCA projects to (n, 2) with 2-element variance array."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 50)).astype(np.float32)
    projected, ev = fit_pca_2d(X)
    assert projected.shape == (60, 2)
    assert ev.shape == (2,)
    assert ev.sum() <= 1.0
    assert np.all(ev >= 0)


def test_fit_tsne_2d_shape():
    """t-SNE returns (n, 2) array."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 50)).astype(np.float32)
    projected = fit_tsne_2d(X, perplexity=5.0)
    assert projected.shape == (60, 2)


def test_compute_silhouette_range():
    """Silhouette score is in [-1, 1] for well-separated clusters."""
    rng = np.random.default_rng(42)
    # Create clearly separated clusters
    cluster_a = rng.standard_normal((30, 10)) + 5.0
    cluster_b = rng.standard_normal((30, 10)) - 5.0
    X = np.vstack([cluster_a, cluster_b]).astype(np.float32)
    labels = ["a"] * 30 + ["b"] * 30
    score = compute_silhouette(X, labels)
    assert -1.0 <= score <= 1.0
    assert score > 0.5  # well-separated clusters should score high


def test_compute_silhouette_single_label():
    """Single label returns 0.0."""
    X = np.random.default_rng(42).standard_normal((20, 10)).astype(np.float32)
    score = compute_silhouette(X, ["same"] * 20)
    assert score == 0.0


def test_explained_variance_report_shape():
    """Returns array of length min(n_components, n_samples, n_features)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 50)).astype(np.float32)
    ev = explained_variance_report(X, n_components=10)
    assert ev.shape == (10,)
    assert np.all(ev >= 0)
    assert ev.sum() <= 1.0


def test_explained_variance_report_caps_components():
    """When n_components > n_features, caps to n_features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5)).astype(np.float32)
    ev = explained_variance_report(X, n_components=20)
    assert len(ev) == 5


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_write_report(tmp_path):
    """Report writes with correct structure."""
    analyses = [
        FeatureAnalysis(
            feature_set="spectral",
            pca_2d=np.zeros((10, 2)),
            tsne_2d=np.zeros((10, 2)),
            silhouette=0.42,
            explained_variance=np.array([0.3, 0.2]),
            labels=["a"] * 5 + ["b"] * 5,
            voices=["ryan"] * 10,
        ),
    ]
    report_path = tmp_path / "report.txt"
    write_report(analyses, report_path)
    text = report_path.read_text()
    assert "spectral" in text
    assert "0.42" in text
    assert "Silhouette score interpretation" in text


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_mood_map_group_help():
    runner = CliRunner()
    result = runner.invoke(main, ["mood-map", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "analyze" in result.output


def test_mood_map_generate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["mood-map", "generate", "--help"])
    assert result.exit_code == 0
    assert "--tts-url" in result.output
    assert "--output" in result.output
    assert "--moods" in result.output
    assert "--voices" in result.output


def test_mood_map_analyze_help():
    runner = CliRunner()
    result = runner.invoke(main, ["mood-map", "analyze", "--help"])
    assert result.exit_code == 0
    assert "--clips" in result.output
    assert "--feature-sets" in result.output
    assert "--tsne-perplexity" in result.output
