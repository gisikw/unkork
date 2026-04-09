"""Voice scoring for optimization. Mel-spectrogram distance + speaker similarity."""

from pathlib import Path

import numpy as np
import soundfile as sf


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def harmonic_mean(values: list[float], weights: list[float] | None = None) -> float:
    """Weighted harmonic mean of positive values. Returns 0 if any value is <= 0.

    When weights are provided: H_w = sum(w_i) / sum(w_i / v_i).
    Without weights, falls back to standard harmonic mean.
    """
    if not values or any(v <= 0 for v in values):
        return 0.0
    if weights is None:
        return len(values) / sum(1.0 / v for v in values)
    if len(weights) != len(values):
        raise ValueError("weights must have same length as values")
    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0
    return w_sum / sum(w / v for w, v in zip(weights, values))


def mel_spectrogram(
    audio: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log mel-spectrogram from audio.

    Uses scipy for STFT and a manual mel filterbank to avoid librosa dependency.

    Returns:
        (n_mels, n_frames) log mel-spectrogram.
    """
    from scipy.signal import stft

    # STFT
    _, _, zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    power = np.abs(zxx) ** 2

    # Mel filterbank
    fbank = _mel_filterbank(sr, n_fft, n_mels)
    mel = fbank @ power

    # Log scale with floor to avoid log(0)
    return np.log(np.maximum(mel, 1e-10))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a mel-scale filterbank matrix."""
    fmin, fmax = 0.0, sr / 2.0
    n_freqs = n_fft // 2 + 1

    # Mel scale conversion
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            fbank[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
        for j in range(bins[i + 1], bins[i + 2]):
            fbank[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)

    return fbank


def mel_spectrogram_distance(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = 24000) -> float:
    """Mean frame-wise cosine distance between mel-spectrograms.

    Uses cosine similarity per frame (column) so the metric measures spectral
    *shape* rather than energy magnitude. This prevents the optimizer from
    gaming the score by flattening spectral energy toward silence.

    Truncates to the shorter signal for alignment. Lower is better (0 = identical shape).
    """
    mel_a = mel_spectrogram(audio_a, sr=sr)
    mel_b = mel_spectrogram(audio_b, sr=sr)

    # Align to shorter
    min_frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_a = mel_a[:, :min_frames]
    mel_b = mel_b[:, :min_frames]

    # Cosine similarity per frame, then average distance
    # Each column is an 80-dim mel vector for one time frame
    similarities = []
    for i in range(min_frames):
        similarities.append(cosine_similarity(mel_a[:, i], mel_b[:, i]))

    return 1.0 - float(np.mean(similarities))


def score_voice_mel(
    generated_audio_paths: list[str],
    reference_audio_paths: list[str],
) -> float:
    """Score generated audio against reference audio using mel-spectrogram distance.

    Each generated audio file is compared to its corresponding reference file.
    Score is 1 / (1 + mean_distance), so higher is better and in [0, 1].

    Args:
        generated_audio_paths: Wav files synthesized with the candidate voice.
        reference_audio_paths: Wav files of the target voice saying the same phrases.

    Returns:
        Score in (0, 1]. Higher is better.
    """
    if not generated_audio_paths or not reference_audio_paths:
        return 0.0

    n = min(len(generated_audio_paths), len(reference_audio_paths))
    distances = []

    for i in range(n):
        gen_audio, gen_sr = sf.read(generated_audio_paths[i])
        ref_audio, ref_sr = sf.read(reference_audio_paths[i])

        # Resample to common rate if needed (both should be 24kHz from Kokoro)
        sr = min(gen_sr, ref_sr)
        distances.append(mel_spectrogram_distance(gen_audio, ref_audio, sr=sr))

    mean_dist = float(np.mean(distances))
    return 1.0 / (1.0 + mean_dist)


def score_voice_resemblyzer(
    generated_audio_paths: list[str],
    reference_audio_paths: list[str],
) -> float:
    """Score generated audio against reference using Resemblyzer speaker similarity.

    Extracts speaker embeddings from each pair, averages, then computes cosine
    similarity. Good at preserving speaker identity / dialect but blind to
    timbre, resonance, and spectral envelope.

    Returns:
        Score in [-1, 1], typically (0, 1]. Higher is better.
    """
    from unkork.embeddings import embed_voice_samples

    if not generated_audio_paths or not reference_audio_paths:
        return 0.0

    n = min(len(generated_audio_paths), len(reference_audio_paths))
    gen_emb = embed_voice_samples(generated_audio_paths[:n])
    ref_emb = embed_voice_samples(reference_audio_paths[:n])
    return cosine_similarity(gen_emb, ref_emb)


def score_voice_composite(
    generated_audio_paths: list[str],
    reference_audio_paths: list[str],
    mel_weight: float = 0.5,
) -> float:
    """Composite score: weighted harmonic mean of mel-spec + Resemblyzer.

    Mel-spectrogram distance captures timbre, resonance, spectral envelope.
    Resemblyzer similarity anchors speaker identity and prevents dialect drift.
    Weighted harmonic mean ensures neither component can be near-zero without
    tanking the composite.

    Args:
        mel_weight: Weight for mel-spectrogram score. Resemblyzer gets (1 - mel_weight).
            Higher values prioritize timbral match; lower values prioritize identity anchor.

    Returns:
        Score in (0, 1]. Higher is better.
    """
    mel_score = score_voice_mel(generated_audio_paths, reference_audio_paths)
    resem_score = score_voice_resemblyzer(generated_audio_paths, reference_audio_paths)

    resem_weight = 1.0 - mel_weight
    return harmonic_mean([mel_score, resem_score], weights=[mel_weight, resem_weight])
