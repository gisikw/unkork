"""Kokoro TTS synthesis wrapper. I/O boundary — requires kokoro-onnx."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Standard phrases for training data generation — diverse phonemes and prosody.
PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "The rain in Spain stays mainly in the plain.",
    "Peter Piper picked a peck of pickled peppers.",
]


def get_pipeline() -> "KPipeline":
    """Load the Kokoro synthesis pipeline. Requires the kokoro package."""
    from kokoro import KPipeline

    return KPipeline(lang_code="a")


def synthesize_voice(
    voice_tensor: torch.Tensor,
    text: str,
    output_path: str | Path,
    pipeline: "KPipeline | None" = None,
) -> Path:
    """Synthesize speech with a voice tensor and save to a wav file.

    Args:
        voice_tensor: (511, 1, 256) Kokoro voice tensor.
        text: Text to synthesize.
        output_path: Where to write the wav file.
        pipeline: Reusable KPipeline instance.

    Returns:
        Path to the written wav file.
    """
    pipeline = pipeline or get_pipeline()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Kokoro expects the voice tensor as a numpy array
    voice_np = voice_tensor.numpy()

    # Generate audio — KPipeline yields (graphemes, phonemes, audio) tuples
    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice_np):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError(f"Kokoro produced no audio for: {text!r}")

    audio = np.concatenate(audio_chunks)
    sf.write(str(output_path), audio, samplerate=24000)
    return output_path


def synthesize_phrases(
    voice_tensor: torch.Tensor,
    output_dir: str | Path,
    phrases: list[str] | None = None,
    pipeline: "KPipeline | None" = None,
) -> list[Path]:
    """Synthesize multiple phrases with a voice tensor.

    Returns:
        List of paths to generated wav files.
    """
    phrases = phrases or PHRASES
    pipeline = pipeline or get_pipeline()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, phrase in enumerate(phrases):
        path = output_dir / f"phrase_{i:03d}.wav"
        synthesize_voice(voice_tensor, phrase, path, pipeline)
        paths.append(path)
    return paths
