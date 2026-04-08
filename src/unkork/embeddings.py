"""Speaker embedding extraction via Resemblyzer."""

from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


def get_encoder() -> VoiceEncoder:
    """Load the Resemblyzer voice encoder (downloads weights on first use)."""
    return VoiceEncoder()


def extract_embedding(audio_path: str | Path, encoder: VoiceEncoder | None = None) -> np.ndarray:
    """Extract a 256-dim speaker embedding from an audio file.

    Returns:
        (256,) float32 array.
    """
    encoder = encoder or get_encoder()
    wav = preprocess_wav(Path(audio_path))
    return encoder.embed_utterance(wav)


def extract_embeddings_batch(
    audio_paths: list[str | Path],
    encoder: VoiceEncoder | None = None,
) -> np.ndarray:
    """Extract speaker embeddings for multiple audio files.

    Returns:
        (n_files, 256) float32 array.
    """
    encoder = encoder or get_encoder()
    embeddings = [extract_embedding(p, encoder) for p in audio_paths]
    return np.stack(embeddings).astype(np.float32)


def embed_voice_samples(
    audio_paths: list[str | Path],
    encoder: VoiceEncoder | None = None,
) -> np.ndarray:
    """Average multiple audio samples into a single speaker embedding.

    Use this when a voice has multiple synthesis samples — averaging
    produces a more stable embedding than any single sample.

    Returns:
        (256,) float32 array.
    """
    embeddings = extract_embeddings_batch(audio_paths, encoder)
    return embeddings.mean(axis=0).astype(np.float32)
