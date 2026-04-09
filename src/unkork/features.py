"""Spectral feature extraction via librosa.

Kitchen-sink approach: extract everything cheap, concatenate into a single
feature vector, let the MLP learn what matters. These capture tonal qualities
(warmth, breathiness, resonance, register) that Resemblyzer deliberately
discards in favor of speaker identity.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def extract_spectral_features(
    audio_path: str | Path,
    sr: int = 24000,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extract spectral features from an audio file.

    Returns a 1-D float32 feature vector with:
        - MFCCs: mean + std per coefficient (n_mfcc * 2)
        - Spectral centroid: mean + std (2)
        - Spectral rolloff: mean + std (2)
        - Spectral bandwidth: mean + std (2)
        - Spectral contrast: mean per 7 bands (7)
        - Zero crossing rate: mean + std (2)
        - RMS energy: mean + std (2)
        - F0 (pyin): mean + std + fraction voiced (3)
        - Mel spectrogram: mean + std per band (80 * 2 = 160)

    Total: n_mfcc*2 + 2 + 2 + 2 + 7 + 2 + 2 + 3 + 160
         = 40 + 2 + 2 + 2 + 7 + 2 + 2 + 3 + 160 = 220 dims (with n_mfcc=20)
    """
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    audio = audio.astype(np.float32)

    features = []

    # MFCCs — capture vocal tract shape, articulation
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features.append(mfccs.mean(axis=1))
    features.append(mfccs.std(axis=1))

    # Spectral centroid — brightness/darkness of voice
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append([centroid.mean(), centroid.std()])

    # Spectral rolloff — where high-frequency energy drops off
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.append([rolloff.mean(), rolloff.std()])

    # Spectral bandwidth — spread of spectral energy
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.append([bandwidth.mean(), bandwidth.std()])

    # Spectral contrast — valley-to-peak ratio per octave band
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.append(contrast.mean(axis=1))  # 7 bands

    # Zero crossing rate — correlates with noisiness/breathiness
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.append([zcr.mean(), zcr.std()])

    # RMS energy — loudness profile
    rms = librosa.feature.rms(y=audio)[0]
    features.append([rms.mean(), rms.std()])

    # F0 via pyin — fundamental frequency (register, pitch)
    # Kept in the combined vector for MLP training, but also extractable
    # separately via extract_f0_features() for independent scoring.
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=50, fmax=600, sr=sr,
    )
    voiced = f0[voiced_flag]
    if len(voiced) > 0:
        features.append([voiced.mean(), voiced.std(), voiced_flag.mean()])
    else:
        features.append([0.0, 0.0, 0.0])

    # Mel spectrogram statistics — spectral envelope shape
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    features.append(log_mel.mean(axis=1))
    features.append(log_mel.std(axis=1))

    return np.concatenate([np.atleast_1d(np.asarray(f, dtype=np.float32)) for f in features])


def extract_spectral_features_batch(
    audio_paths: list[str | Path],
    sr: int = 24000,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extract spectral features from multiple audio files and average.

    Same as embed_voice_samples in embeddings.py — averaging across
    multiple samples of the same voice gives a more stable feature vector.

    Returns:
        (feature_dim,) float32 array.
    """
    features = [extract_spectral_features(p, sr=sr, n_mfcc=n_mfcc) for p in audio_paths]
    return np.stack(features).mean(axis=0).astype(np.float32)


def extract_f0_features(
    audio_path: str | Path,
    sr: int = 24000,
) -> np.ndarray:
    """Extract F0/pitch features only.

    Returns a 1-D float32 vector with:
        - F0 mean (Hz) — register / pitch center
        - F0 std (Hz) — pitch variability / expressiveness
        - Fraction voiced — how much of the signal is pitched

    These are the same 3 values embedded in the full spectral feature vector,
    but extracted independently for separate scoring. This lets refinement
    target register without flattening pitch dynamics or tonal character.
    """
    audio, file_sr = sf.read(audio_path)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    audio = audio.astype(np.float32)

    f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=600, sr=sr)
    voiced = f0[voiced_flag]
    if len(voiced) > 0:
        return np.array([voiced.mean(), voiced.std(), voiced_flag.mean()], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def extract_f0_features_batch(
    audio_paths: list[str | Path],
    sr: int = 24000,
) -> np.ndarray:
    """Extract and average F0 features across multiple audio files.

    Returns:
        (3,) float32 array.
    """
    features = [extract_f0_features(p, sr=sr) for p in audio_paths]
    return np.stack(features).mean(axis=0).astype(np.float32)
