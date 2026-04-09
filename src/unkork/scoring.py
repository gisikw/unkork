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


def harmonic_mean(values: list[float]) -> float:
    """Harmonic mean of positive values. Returns 0 if any value is <= 0."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


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
    """Mean squared error between mel-spectrograms of two audio signals.

    Truncates to the shorter signal for alignment. Lower is better.
    """
    mel_a = mel_spectrogram(audio_a, sr=sr)
    mel_b = mel_spectrogram(audio_b, sr=sr)

    # Align to shorter
    min_frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_a = mel_a[:, :min_frames]
    mel_b = mel_b[:, :min_frames]

    return float(np.mean((mel_a - mel_b) ** 2))


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
