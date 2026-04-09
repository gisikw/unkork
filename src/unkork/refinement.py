"""CMA-ES refinement of voice tensors in PCA space."""

import tempfile
from pathlib import Path

import cma
import click
import numpy as np
import torch

from unkork.embeddings import extract_embedding, get_encoder
from unkork.pca import PCATransform
from unkork.pca import inverse_transform as pca_inverse
from unkork.pca import transform as pca_forward
from unkork.scoring import score_voice
from unkork.synthesis import get_pipeline, synthesize_phrases
from unkork.tensors import flatten, unflatten


def refine_tensor(
    start_tensor: torch.Tensor,
    target_audio: str | Path,
    pca: PCATransform,
    max_iterations: int = 50,
    popsize: int = 8,
    sigma0: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Refine a voice tensor toward a target voice using CMA-ES in PCA space.

    Optimizes in the PCA-projected space (50 dims) rather than the full
    voice tensor space (130k dims), keeping CMA-ES's covariance matrix
    tractable.

    Args:
        start_tensor: Starting voice tensor (e.g., from MLP prediction).
        target_audio: Path to target reference audio.
        pca: Fitted PCA transform for projecting to/from voice space.
        max_iterations: Maximum CMA-ES generations.
        popsize: Population size per generation.
        sigma0: Initial step size in PCA space.

    Returns:
        (refined_tensor, best_score)
    """
    encoder = get_encoder()
    pipeline = get_pipeline()

    target_embedding = extract_embedding(str(target_audio), encoder)

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
                paths = synthesize_phrases(tensor, tmpdir, pipeline=pipeline)
                audio_paths = [str(p) for p in paths]
                score = score_voice(audio_paths, target_embedding, encoder)
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
            "verbose": -1,  # suppress CMA-ES output; we log ourselves
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
