"""CLI for unkork — regression codec for Kokoro TTS voice creation."""

from pathlib import Path

import click
import numpy as np
import torch


@click.group()
def main():
    """unkork — regression codec for Kokoro TTS voice creation."""
    pass


@main.command()
@click.option("--voices-dir", required=True, help="Path to Kokoro voice .pt files")
@click.option("--output", default="data/", help="Output directory for training data")
@click.option("--count", default=500, help="Number of synthetic voices to generate")
@click.option("--phrases", default=5, help="Phrases to synthesize per voice")
def generate(voices_dir: str, output: str, count: int, phrases: int):
    """Generate training dataset from existing Kokoro voices.

    1. Load all .pt voice files from --voices-dir
    2. Generate --count random blends
    3. Synthesize --phrases phrases per blend via Kokoro
    4. Extract features: Resemblyzer embedding (256-dim) + librosa spectral
       features (~220-dim), concatenated into a single input vector
    5. Save (features, flat tensors) to --output
    """
    from unkork.dataset import save_dataset
    from unkork.embeddings import embed_voice_samples, get_encoder
    from unkork.features import extract_spectral_features_batch
    from unkork.synthesis import PHRASES, get_pipeline, synthesize_phrases
    from unkork.tensors import flatten, load_voice, random_blends

    # Load source voices
    voice_dir = Path(voices_dir)
    pt_files = sorted(voice_dir.glob("*.pt"))
    if len(pt_files) < 2:
        raise click.ClickException(f"Need at least 2 voice files, found {len(pt_files)}")
    click.echo(f"Loaded {len(pt_files)} source voices from {voices_dir}")

    source_tensors = [load_voice(p) for p in pt_files]

    # Generate random blends
    click.echo(f"Generating {count} random blends...")
    blends = random_blends(source_tensors, count)

    # Synthesize and extract features
    click.echo("Synthesizing and extracting features...")
    pipeline = get_pipeline()
    encoder = get_encoder()
    phrase_subset = PHRASES[:phrases]

    output_dir = Path(output)
    audio_dir = output_dir / "audio"

    all_features = []
    all_flat_tensors = []

    with click.progressbar(enumerate(blends), length=count, label="Processing") as bar:
        for i, tensor in bar:
            voice_audio_dir = audio_dir / f"voice_{i:04d}"
            paths = synthesize_phrases(tensor, voice_audio_dir, phrase_subset, pipeline)
            str_paths = [str(p) for p in paths]

            # Resemblyzer embedding (256-dim)
            resem_embedding = embed_voice_samples(str_paths, encoder)
            # Librosa spectral features (~220-dim)
            spectral_features = extract_spectral_features_batch(str_paths)
            # Concatenate into single input vector
            combined = np.concatenate([resem_embedding, spectral_features])

            all_features.append(combined)
            all_flat_tensors.append(flatten(tensor))

    features_arr = np.stack(all_features)
    tensors_arr = np.stack(all_flat_tensors)

    save_dataset(features_arr, tensors_arr, output_dir)
    click.echo(f"Saved dataset: {features_arr.shape[0]} samples, {features_arr.shape[1]}-dim features -> {output_dir}")


@main.command()
@click.option("--data", default="data/", help="Training data directory")
@click.option("--output", default="models/", help="Output directory for trained model")
@click.option("--epochs", default=200, help="Training epochs")
@click.option("--n-components", default=50, help="PCA components to keep")
@click.option("--hidden-dim", default=512, help="MLP hidden layer dimension")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--batch-size", default=32, help="Batch size")
def train(
    data: str,
    output: str,
    epochs: int,
    n_components: int,
    hidden_dim: int,
    lr: float,
    batch_size: int,
):
    """Train the regression codec MLP.

    1. Load dataset from --data
    2. Fit PCA on voice tensors (--n-components dimensions)
    3. Project tensors to PCA space
    4. Train MLP: input_dim -> hidden -> hidden -> n_components
    5. Save model checkpoint + PCA transform to --output

    Input dimension is determined by the dataset (256 for Resemblyzer-only,
    ~476 for augmented features with librosa spectral features).
    """
    from unkork.dataset import load_dataset
    from unkork.model import VoiceCodec, save_checkpoint
    from unkork.pca import fit as fit_pca
    from unkork.pca import save as save_pca
    from unkork.pca import transform as pca_transform
    from unkork.training import build_datasets, train_codec

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    click.echo(f"Loading dataset from {data}...")
    embeddings, flat_tensors = load_dataset(data)
    click.echo(f"  {embeddings.shape[0]} samples, embeddings {embeddings.shape[1]}-dim")

    # Fit PCA
    click.echo(f"Fitting PCA with {n_components} components...")
    pca = fit_pca(flat_tensors, n_components)
    explained = pca.explained_variance_ratio.sum()
    click.echo(f"  Explained variance: {explained:.1%}")

    pca_path = output_dir / "pca.npz"
    save_pca(pca, pca_path)
    click.echo(f"  Saved PCA transform -> {pca_path}")

    # Project to PCA space
    targets = pca_transform(pca, flat_tensors)

    # Build datasets
    train_ds, val_ds = build_datasets(embeddings, targets)
    click.echo(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f"Training on {device} for {epochs} epochs...")

    model = VoiceCodec(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        output_dim=n_components,
    )

    result = train_codec(
        model, train_ds, val_ds,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device,
    )

    click.echo(f"  Best val loss: {result.best_val_loss:.6f} at epoch {result.best_epoch}")

    # Save model
    model_path = output_dir / "codec.pt"
    save_checkpoint(
        model, str(model_path),
        n_components=n_components,
        hidden_dim=hidden_dim,
        best_val_loss=result.best_val_loss,
        best_epoch=result.best_epoch,
    )
    click.echo(f"  Saved model -> {model_path}")
    click.echo("Done.")


@main.command()
@click.option("--model", required=True, help="Path to trained codec model (.pt)")
@click.option("--pca", required=True, help="Path to PCA transform (.npz)")
@click.option("--audio", required=True, help="Reference audio file")
@click.option("--output", default="voice.pt", help="Output voice tensor path")
def predict(model: str, pca: str, audio: str, output: str):
    """Predict a Kokoro voice tensor from reference audio.

    1. Extract features from --audio (Resemblyzer + librosa spectral)
    2. Run through trained MLP
    3. Inverse PCA transform -> full voice tensor
    4. Save as .pt
    """
    import torch

    from unkork.embeddings import extract_embedding
    from unkork.features import extract_spectral_features
    from unkork.model import load_checkpoint
    from unkork.pca import inverse_transform
    from unkork.pca import load as load_pca
    from unkork.tensors import save_voice, unflatten

    click.echo(f"Loading model from {model}...")
    codec, metadata = load_checkpoint(model)
    codec.eval()

    click.echo(f"Loading PCA from {pca}...")
    pca_transform = load_pca(pca)

    click.echo(f"Extracting features from {audio}...")
    resem_embedding = extract_embedding(audio)
    spectral_features = extract_spectral_features(audio)
    features = np.concatenate([resem_embedding, spectral_features])
    click.echo(f"  Feature vector: {features.shape[0]}-dim (resemblyzer={resem_embedding.shape[0]}, spectral={spectral_features.shape[0]})")

    # Predict PCA components
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        pca_pred = codec(x).squeeze(0).numpy()

    # Inverse PCA -> flat tensor -> voice tensor
    flat = inverse_transform(pca_transform, pca_pred)
    tensor = unflatten(flat.astype(np.float32))

    save_voice(tensor, output)
    click.echo(f"Saved predicted voice tensor -> {output}")


@main.command()
@click.option("--start", required=True, help="Starting voice tensor (.pt)")
@click.option("--reference-dir", required=True, help="Directory of target reference wav files")
@click.option("--pca", required=True, help="Path to PCA transform (.npz)")
@click.option("--iterations", default=50, help="CMA-ES iterations")
@click.option("--popsize", default=8, help="CMA-ES population size per generation")
@click.option("--sigma", default=0.5, help="CMA-ES initial step size in PCA space")
@click.option("--output", default="refined.pt", help="Output refined tensor path")
@click.option(
    "--scorer", default="composite",
    type=click.Choice(["mel", "resemblyzer", "composite"]),
    help="Scoring function: mel (timbre), resemblyzer (identity), composite (both)",
)
@click.option(
    "--mel-weight", default=0.5, type=float,
    help="Mel-spec weight in composite scorer (0-1). Resemblyzer gets the rest.",
)
def refine(
    start: str, reference_dir: str, pca: str, iterations: int,
    popsize: int, sigma: float, output: str, scorer: str, mel_weight: float,
):
    """Refine a voice tensor toward target audio using CMA-ES optimization.

    Supports three scoring modes:

    \b
      mel        — Mel-spectrogram distance only (timbre, resonance, spectral envelope)
      resemblyzer — Resemblyzer speaker similarity only (identity, dialect)
      composite  — Weighted harmonic mean of both (default)

    For composite mode, --mel-weight controls the balance. Higher values
    prioritize timbral match; lower values anchor dialect/identity.
    """
    from functools import partial

    from unkork.pca import load as load_pca
    from unkork.refinement import refine_tensor
    from unkork.scoring import score_voice_composite, score_voice_mel, score_voice_resemblyzer
    from unkork.tensors import load_voice, save_voice

    # Build scoring function
    scorers = {
        "mel": score_voice_mel,
        "resemblyzer": score_voice_resemblyzer,
        "composite": partial(score_voice_composite, mel_weight=mel_weight),
    }
    score_fn = scorers[scorer]

    click.echo(f"Loading starting tensor from {start}...")
    start_tensor = load_voice(start)

    click.echo(f"Loading PCA from {pca}...")
    pca_transform = load_pca(pca)

    ref_dir = Path(reference_dir)
    ref_paths = sorted(str(p) for p in ref_dir.glob("*.wav"))
    if not ref_paths:
        raise click.ClickException(f"No .wav files found in {reference_dir}")
    click.echo(f"Found {len(ref_paths)} reference audio files")

    scorer_desc = scorer
    if scorer == "composite":
        scorer_desc = f"composite (mel={mel_weight:.1f}, resemblyzer={1-mel_weight:.1f})"
    click.echo(f"Scorer: {scorer_desc}")
    click.echo(f"Refining ({iterations} iterations, pop={popsize})...")
    refined, score = refine_tensor(
        start_tensor, ref_paths, pca_transform,
        max_iterations=iterations, popsize=popsize, sigma0=sigma,
        scorer=score_fn,
    )

    save_voice(refined, output)
    click.echo(f"Saved refined tensor -> {output} (score: {score:.4f})")


@main.command()
@click.option("--voice", required=True, help="Voice tensor (.pt) to test")
@click.option("--text", default="Hello Kevin. This is what I sound like in Kokoro.", help="Text to speak")
@click.option("--output", default="test.wav", help="Output wav file path")
def speak(voice: str, text: str, output: str):
    """Synthesize speech with a voice tensor to test how it sounds."""
    from unkork.synthesis import get_pipeline, synthesize_voice
    from unkork.tensors import load_voice

    click.echo(f"Loading voice from {voice}...")
    tensor = load_voice(voice)

    click.echo(f"Synthesizing: {text!r}")
    synthesize_voice(tensor, text, output, get_pipeline())
    click.echo(f"Wrote {output}")
