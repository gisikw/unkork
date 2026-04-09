"""CMA-ES refinement of voice tensors in PCA space."""

import tempfile
from pathlib import Path

import cma
import click
import numpy as np
import torch

from unkork.pca import PCATransform
from unkork.pca import inverse_transform as pca_inverse
from unkork.pca import transform as pca_forward
from unkork.scoring import score_voice_mel
from unkork.synthesis import PHRASES, get_pipeline, synthesize_phrases
from unkork.tensors import flatten, unflatten


def refine_tensor(
    start_tensor: torch.Tensor,
    reference_audio_paths: list[str],
    pca: PCATransform,
    phrases: list[str] | None = None,
    max_iterations: int = 50,
    popsize: int = 8,
    sigma0: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Refine a voice tensor toward target audio using CMA-ES in PCA space.

    Scores candidates by mel-spectrogram distance against reference audio,
    capturing timbre, resonance, and spectral envelope — not just speaker
    identity.

    Args:
        start_tensor: Starting voice tensor (e.g., from MLP prediction).
        reference_audio_paths: Wav files of the target voice speaking the
            same phrases used for synthesis (one per phrase).
        pca: Fitted PCA transform for projecting to/from voice space.
        phrases: Text phrases to synthesize for comparison. Must match
            the reference audio. Defaults to PHRASES[:len(reference_audio_paths)].
        max_iterations: Maximum CMA-ES generations.
        popsize: Population size per generation.
        sigma0: Initial step size in PCA space.

    Returns:
        (refined_tensor, best_score)
    """
    pipeline = get_pipeline()

    phrases = phrases or PHRASES[: len(reference_audio_paths)]
    ref_paths = reference_audio_paths[: len(phrases)]

    # Project starting tensor into PCA space
    flat = flatten(start_tensor).astype(np.float64)
    x0 = pca_forward(pca, flat)

    best_score = 0.0
    best_x = x0.copy()

    def objective(x: np.ndarray) -> float:
        nonlocal best_score, best_x

        # Unproject from PCA space to full voice tensor
        flat_voice = pca_inverse(pca, x).astype(np.float32)
        tensor = unflatten(flat_voice)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                paths = synthesize_phrases(tensor, tmpdir, phrases, pipeline)
                gen_paths = [str(p) for p in paths]
                score = score_voice_mel(gen_paths, ref_paths)
            except Exception:
                return 1.0  # worst possible (CMA-ES minimizes)

        if score > best_score:
            best_score = score
            best_x = x.copy()

        return -score  # negate: CMA-ES minimizes

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "maxiter": max_iterations,
            "popsize": popsize,
            "verbose": -1,
        },
    )

    generation = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        generation += 1
        click.echo(f"  gen {generation}: best score = {best_score:.4f}")

    # Unproject best result back to voice tensor
    best_flat = pca_inverse(pca, best_x).astype(np.float32)
    result_tensor = unflatten(best_flat)
    return result_tensor, best_score
