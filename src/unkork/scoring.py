"""Voice scoring for optimization. Harmonic mean of similarity signals."""

import numpy as np
from resemblyzer import VoiceEncoder

from unkork.embeddings import embed_voice_samples, extract_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean of positive values. Returns 0 if any value is <= 0."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def score_voice(
    generated_audio_paths: list[str],
    target_embedding: np.ndarray,
    encoder: VoiceEncoder | None = None,
) -> float:
    """Score a generated voice against a target speaker embedding.

    Uses a harmonic mean of:
    1. Resemblyzer similarity — how close the voice is to the target
    2. Self-similarity — how consistent the voice is across phrases

    The harmonic mean penalizes any single low signal, preventing
    degenerate solutions that fool one metric but not others.

    Args:
        generated_audio_paths: Paths to wav files synthesized with the candidate voice.
        target_embedding: (256,) target speaker embedding.
        encoder: Reusable VoiceEncoder instance.

    Returns:
        Score in [0, 1]. Higher is better.
    """
    from unkork.embeddings import get_encoder

    encoder = encoder or get_encoder()

    if not generated_audio_paths:
        return 0.0

    # Extract embeddings for each generated phrase
    gen_embeddings = [extract_embedding(p, encoder) for p in generated_audio_paths]

    # Signal 1: Resemblyzer similarity to target (average across phrases)
    target_sims = [cosine_similarity(e, target_embedding) for e in gen_embeddings]
    resemblyzer_score = float(np.mean(target_sims))

    # Signal 2: Self-similarity (average pairwise similarity between phrases)
    if len(gen_embeddings) < 2:
        self_sim = 1.0
    else:
        pairwise = []
        for i in range(len(gen_embeddings)):
            for j in range(i + 1, len(gen_embeddings)):
                pairwise.append(cosine_similarity(gen_embeddings[i], gen_embeddings[j]))
        self_sim = float(np.mean(pairwise))

    # Shift similarities from [-1, 1] to [0, 1] for harmonic mean
    resemblyzer_shifted = (resemblyzer_score + 1) / 2
    self_sim_shifted = (self_sim + 1) / 2

    return harmonic_mean([resemblyzer_shifted, self_sim_shifted])
