"""PCA mood mapping experiment.

Generates mood-conditioned audio clips via Qwen3-TTS, extracts acoustic
features, and analyzes whether mood categories form separable clusters
in feature space. If they do, mood-controlled voice interpolation is viable.

Two-phase pipeline:
  1. generate — call Qwen3-TTS with mood instructions, save organized WAVs
  2. analyze  — extract features per clip, run PCA/t-SNE, compute silhouette
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


MOODS = ["sultry", "assertive", "tender", "irritable", "playful", "neutral"]

MOOD_INSTRUCTIONS = {
    "sultry": "Speak slowly and warmly, with a low, smooth, seductive tone.",
    "assertive": "Speak clearly and confidently, with authority and a direct, steady tone.",
    "tender": "Speak softly and gently, with warmth and care, as if comforting someone.",
    "irritable": "Speak with a clipped, slightly impatient edge, as if mildly annoyed.",
    "playful": "Speak lightly and playfully, with a teasing energy and varied pace.",
    "neutral": "Speak naturally and evenly, without any particular emotional emphasis.",
}

# Emotionally neutral sentences — meaning comes from delivery, not content.
SENTENCES = [
    "The afternoon light fell across the empty chair.",
    "She counted the coins twice before putting them away.",
    "He opened the window and looked out at the street.",
    "The package arrived three days later than expected.",
    "They finished the meal without saying much.",
    "A single cloud drifted past the face of the moon.",
    "The machine made a low humming sound and stopped.",
    "He checked his watch and put it back in his pocket.",
    "The road curved sharply to the left before the bridge.",
    "She left the door unlocked when she went inside.",
    "The letter was written in a careful, even hand.",
    "Rain gathered in the gutters and began to flow.",
    "He folded the paper once and set it on the table.",
    "The town had changed since she last visited.",
    "Water dripped from the faucet at irregular intervals.",
    "He read the paragraph again and set the book down.",
    "The light in the hallway flickered once, then held.",
    "She noted the time and continued with her work.",
    "The box was heavier than it appeared from a distance.",
    "He turned off the lamp and waited in the dark.",
]

MOOD_COLORS = {
    "sultry": "#c0392b",
    "assertive": "#2980b9",
    "tender": "#27ae60",
    "irritable": "#e67e22",
    "playful": "#8e44ad",
    "neutral": "#7f8c8d",
}

VOICE_MARKERS = {
    "ryan": "o",
    "aiden": "s",
}


@dataclass
class ClipRecord:
    """Metadata for one generated audio clip."""

    path: str
    mood: str
    voice: str
    sentence_idx: int


@dataclass
class FeatureAnalysis:
    """Results for one feature set."""

    feature_set: str
    pca_2d: np.ndarray
    tsne_2d: np.ndarray
    silhouette: float
    explained_variance: np.ndarray
    labels: list[str]
    voices: list[str]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def mood_instruction(mood: str) -> str:
    """Return the Qwen3-TTS instruction string for a mood."""
    if mood not in MOOD_INSTRUCTIONS:
        raise ValueError(f"Unknown mood {mood!r}, expected one of {list(MOOD_INSTRUCTIONS)}")
    return MOOD_INSTRUCTIONS[mood]


def build_tts_request(
    text: str,
    mood: str,
    voice: str,
    instructions: dict[str, str] | None = None,
) -> dict:
    """Build the JSON body for a Qwen3-TTS /v1/audio/speech request."""
    inst = (instructions or MOOD_INSTRUCTIONS).get(mood)
    if inst is None:
        inst = mood_instruction(mood)
    return {
        "model": "qwen3-tts",
        "voice": voice,
        "input": text,
        "response_format": "wav",
        "instructions": inst,
    }


MIN_CLIP_BYTES = 10_000  # ~200ms at 24kHz 16-bit mono; anything smaller is suspect


def generate_clip(
    tts_url: str,
    text: str,
    mood: str,
    voice: str,
    output_path: Path,
    instructions: dict[str, str] | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
) -> Path:
    """Call Qwen3-TTS and write audio to output_path.

    Validates response content-type and minimum size to catch error pages
    or truncated responses being saved as .wav files.
    """
    import requests

    body = build_tts_request(text, mood, voice, instructions)
    resp = requests.post(
        f"{tts_url.rstrip('/')}/v1/audio/speech",
        json=body,
        timeout=timeout,
        verify=verify_ssl,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"TTS request failed ({resp.status_code}): {resp.text[:500]}"
        )
    content_type = resp.headers.get("content-type", "")
    if "audio" not in content_type and "octet-stream" not in content_type:
        raise RuntimeError(
            f"TTS returned non-audio content-type {content_type!r}: {resp.text[:500]}"
        )
    if len(resp.content) < MIN_CLIP_BYTES:
        raise RuntimeError(
            f"TTS returned suspiciously small response ({len(resp.content)} bytes, "
            f"expected >={MIN_CLIP_BYTES}). Content-type: {content_type}. "
            f"First 200 bytes: {resp.content[:200]!r}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    return output_path


def generate_clips(
    tts_url: str,
    output_dir: Path,
    moods: list[str],
    voices: list[str],
    sentences: list[str],
    instructions: dict[str, str] | None = None,
    timeout: int = 60,
    verify_ssl: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[ClipRecord]:
    """Generate the full clip matrix and write manifest.json.

    Idempotent: skips clips that already exist on disk.
    """
    records: list[ClipRecord] = []
    total = len(moods) * len(voices) * len(sentences)
    done = 0

    for mood in moods:
        for voice in voices:
            for idx, sentence in enumerate(sentences):
                clip_path = output_dir / mood / f"{voice}_{idx:03d}.wav"
                if not clip_path.exists():
                    generate_clip(
                        tts_url, sentence, mood, voice, clip_path,
                        instructions=instructions, timeout=timeout,
                        verify_ssl=verify_ssl,
                    )
                records.append(ClipRecord(
                    path=str(clip_path),
                    mood=mood,
                    voice=voice,
                    sentence_idx=idx,
                ))
                done += 1
                if on_progress:
                    on_progress(done, total)

    save_manifest(records, output_dir / "manifest.json")
    return records


def save_manifest(records: list[ClipRecord], path: Path) -> None:
    """Write clip records to JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def load_manifest(path: Path) -> list[ClipRecord]:
    """Read manifest.json and reconstruct ClipRecord list."""
    data = json.loads(path.read_text())
    return [ClipRecord(**r) for r in data]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features_for_clips(
    records: list[ClipRecord],
    feature_set: str,
    encoder=None,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Extract features for all clips. Returns (n_clips, feature_dim) array.

    feature_set: "resemblyzer" | "spectral" | "combined" | "f0"
    encoder: optional pre-loaded VoiceEncoder (avoids reloading per call)
    """
    from unkork.embeddings import extract_embedding, get_encoder
    from unkork.features import extract_f0_features, extract_spectral_features

    if feature_set not in ("resemblyzer", "spectral", "combined", "f0"):
        raise ValueError(f"Unknown feature_set {feature_set!r}")

    if feature_set in ("resemblyzer", "combined") and encoder is None:
        encoder = get_encoder()

    features = []
    total = len(records)
    for i, rec in enumerate(records):
        if feature_set == "resemblyzer":
            vec = extract_embedding(rec.path, encoder)
        elif feature_set == "spectral":
            vec = extract_spectral_features(rec.path)
        elif feature_set == "f0":
            vec = extract_f0_features(rec.path)
        elif feature_set == "combined":
            resem = extract_embedding(rec.path, encoder)
            spectral = extract_spectral_features(rec.path)
            vec = np.concatenate([resem, spectral])
        features.append(vec)
        if on_progress:
            on_progress(i + 1, total)

    return np.stack(features).astype(np.float32)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def fit_pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA projection to 2D. Returns (projected, explained_variance_ratio[:2])."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    projected = pca.fit_transform(features)
    return projected, pca.explained_variance_ratio_


def fit_tsne_2d(
    features: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """t-SNE projection to 2D. Returns (n_samples, 2) array."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(features)


def compute_silhouette(features: np.ndarray, labels: list[str]) -> float:
    """Silhouette score for mood clusters. Returns float in [-1, 1]."""
    from sklearn.metrics import silhouette_score

    unique = set(labels)
    if len(unique) < 2:
        return 0.0
    return float(silhouette_score(features, labels, metric="cosine"))


def explained_variance_report(features: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Fit PCA with n_components and return explained_variance_ratio array."""
    from sklearn.decomposition import PCA

    n = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n)
    pca.fit(features)
    return pca.explained_variance_ratio_


def analyze_feature_set(
    records: list[ClipRecord],
    feature_set: str,
    tsne_perplexity: float = 30.0,
    n_variance_components: int = 10,
    encoder=None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FeatureAnalysis:
    """Run full analysis pipeline for one feature set."""
    features = extract_features_for_clips(
        records, feature_set, encoder=encoder, on_progress=on_progress,
    )
    pca_2d, _ = fit_pca_2d(features)
    tsne_2d = fit_tsne_2d(features, perplexity=tsne_perplexity)
    labels = [r.mood for r in records]
    sil = compute_silhouette(features, labels)
    ev = explained_variance_report(features, n_variance_components)

    return FeatureAnalysis(
        feature_set=feature_set,
        pca_2d=pca_2d,
        tsne_2d=tsne_2d,
        silhouette=sil,
        explained_variance=ev,
        labels=labels,
        voices=[r.voice for r in records],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scatter(
    projected: np.ndarray,
    labels: list[str],
    voices: list[str],
    title: str,
    output_path: Path,
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
) -> None:
    """Save a 2D scatter plot colored by mood, shaped by voice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or MOOD_COLORS
    markers = markers or VOICE_MARKERS
    default_marker = "^"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each mood+voice combination for proper legend
    moods_seen = set()
    voices_seen = set()
    for mood in dict.fromkeys(labels):  # preserves order, deduplicates
        for voice in dict.fromkeys(voices):
            mask = [(l == mood and v == voice) for l, v in zip(labels, voices)]
            if not any(mask):
                continue
            pts = projected[mask]
            c = colors.get(mood, "#333333")
            m = markers.get(voice, default_marker)
            label_str = f"{mood}" if voice not in voices_seen or mood not in moods_seen else None
            # Only label once per mood
            label_str = f"{mood}" if mood not in moods_seen else None
            ax.scatter(pts[:, 0], pts[:, 1], c=c, marker=m, alpha=0.7,
                       s=40, label=label_str, edgecolors="white", linewidths=0.3)
            moods_seen.add(mood)
            voices_seen.add(voice)

    # Add voice legend entries
    unique_voices = list(dict.fromkeys(voices))
    if len(unique_voices) > 1:
        for voice in unique_voices:
            m = markers.get(voice, default_marker)
            ax.scatter([], [], c="gray", marker=m, s=40, label=f"voice: {voice}")

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_explained_variance(
    explained_variance: np.ndarray,
    feature_set: str,
    output_path: Path,
) -> None:
    """Save a bar chart of explained variance per PC (scree plot)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(explained_variance)
    ax.bar(range(1, n + 1), explained_variance, color="#2980b9", alpha=0.8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"PCA Scree Plot — {feature_set}")
    ax.set_xticks(range(1, n + 1))
    cumulative = np.cumsum(explained_variance)
    ax.plot(range(1, n + 1), cumulative, "ro-", markersize=4, label="Cumulative")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(analyses: list[FeatureAnalysis], output_path: Path) -> None:
    """Write a plain-text summary of silhouette scores and variance."""
    lines = ["Mood Mapping Analysis Report", "=" * 40, ""]
    lines.append(f"{'Feature Set':<20} {'Silhouette':>12} {'PC1 Var':>10} {'PC2 Var':>10}")
    lines.append("-" * 55)
    for a in analyses:
        pc1 = a.explained_variance[0] if len(a.explained_variance) > 0 else 0.0
        pc2 = a.explained_variance[1] if len(a.explained_variance) > 1 else 0.0
        lines.append(f"{a.feature_set:<20} {a.silhouette:>12.4f} {pc1:>10.4f} {pc2:>10.4f}")
    lines.append("")
    lines.append("Silhouette score interpretation:")
    lines.append("  > 0.5  — strong clustering (mood clearly separates)")
    lines.append("  0.2-0.5 — moderate clustering (mood influence detectable)")
    lines.append("  0.0-0.2 — weak clustering (mood barely separates)")
    lines.append("  < 0.0  — no clustering (overlapping, random)")
    lines.append("")

    n_clips = len(analyses[0].labels) if analyses else 0
    n_moods = len(set(analyses[0].labels)) if analyses else 0
    n_voices = len(set(analyses[0].voices)) if analyses else 0
    lines.append(f"Clips: {n_clips}  Moods: {n_moods}  Voices: {n_voices}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
