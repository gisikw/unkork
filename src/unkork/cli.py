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
    type=click.Choice(["mel", "resemblyzer", "spectral", "f0", "composite"]),
    help="Scoring function: mel, resemblyzer, spectral, f0, or composite (all four)",
)
@click.option("--mel-weight", default=0.30, type=float, help="Mel-spec weight in composite scorer")
@click.option("--spectral-weight", default=0.30, type=float, help="Spectral features weight in composite scorer")
@click.option("--f0-weight", default=0.15, type=float, help="F0/register weight in composite scorer")
@click.option("--resemblyzer-weight", default=0.25, type=float, help="Resemblyzer weight in composite scorer")
def refine(
    start: str, reference_dir: str, pca: str, iterations: int,
    popsize: int, sigma: float, output: str, scorer: str,
    mel_weight: float, spectral_weight: float, f0_weight: float, resemblyzer_weight: float,
):
    """Refine a voice tensor toward target audio using CMA-ES optimization.

    Supports five scoring modes:

    \b
      mel        — Mel-spectrogram cosine distance (spectral shape per frame)
      resemblyzer — Resemblyzer speaker similarity (identity, dialect anchor)
      spectral   — Librosa spectral features (MFCCs, resonance, warmth)
      f0         — F0/pitch only (register match without flattening dynamics)
      composite  — Weighted harmonic mean of all four (default)

    For composite mode, --mel-weight, --spectral-weight, --f0-weight, and
    --resemblyzer-weight control the balance. Weights are relative.
    """
    from functools import partial

    from unkork.pca import load as load_pca
    from unkork.refinement import refine_tensor
    from unkork.scoring import (
        score_voice_composite,
        score_voice_f0,
        score_voice_mel,
        score_voice_resemblyzer,
        score_voice_spectral,
    )
    from unkork.tensors import load_voice, save_voice

    # Build scoring function
    scorers = {
        "mel": score_voice_mel,
        "resemblyzer": score_voice_resemblyzer,
        "spectral": score_voice_spectral,
        "f0": score_voice_f0,
        "composite": partial(
            score_voice_composite,
            mel_weight=mel_weight,
            spectral_weight=spectral_weight,
            f0_weight=f0_weight,
            resemblyzer_weight=resemblyzer_weight,
        ),
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
        scorer_desc = f"composite (mel={mel_weight:.2f}, spectral={spectral_weight:.2f}, f0={f0_weight:.2f}, resemblyzer={resemblyzer_weight:.2f})"
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
@click.option("--text", default="Hello. This is what I sound like in Kokoro.", help="Text to speak")
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


# ---------------------------------------------------------------------------
# mood-map experiment
# ---------------------------------------------------------------------------


@main.group("mood-map")
def mood_map():
    """Mood mapping experiment: does emotional style separate in feature space?"""
    pass


@mood_map.command("generate")
@click.option(
    "--tts-url", default="http://localhost:8880",
    help="Base URL for Qwen3-TTS (POST /v1/audio/speech)",
)
@click.option("--output", required=True, help="Directory for audio clips + manifest.json")
@click.option(
    "--moods", default="sultry,assertive,tender,irritable,playful,neutral",
    help="Comma-separated mood list",
)
@click.option("--voices", default="ryan,aiden", help="Comma-separated Qwen3-TTS voice list")
@click.option("--timeout", default=60, type=int, help="HTTP timeout per request (seconds)")
def mood_map_generate(tts_url: str, output: str, moods: str, voices: str, timeout: int):
    """Generate mood-conditioned audio clips via Qwen3-TTS.

    Calls POST /v1/audio/speech with per-mood instruction strings.
    Writes one wav per (mood, voice, sentence) plus manifest.json.

    \b
    Idempotent: skips clips that already exist on disk.

    \b
    Container networking — TTS is typically on the host at localhost:8880.
    Either run with --network=host or use the bridge gateway IP:
        unkork mood-map generate --tts-url http://10.88.0.1:8880 --output /data/mood-clips
    """
    from unkork.mood_map import SENTENCES, generate_clips

    output_dir = Path(output)
    mood_list = [m.strip() for m in moods.split(",")]
    voice_list = [v.strip() for v in voices.split(",")]

    total = len(mood_list) * len(voice_list) * len(SENTENCES)
    click.echo(f"Generating {total} clips: {len(mood_list)} moods × {len(voice_list)} voices × {len(SENTENCES)} sentences")
    click.echo(f"TTS: {tts_url}")
    click.echo(f"Output: {output_dir}")

    def on_progress(done: int, total: int) -> None:
        click.echo(f"  [{done}/{total}] clips", nl=(done == total))

    records = generate_clips(
        tts_url, output_dir, mood_list, voice_list, SENTENCES,
        timeout=timeout, on_progress=on_progress,
    )
    click.echo(f"Done. {len(records)} clips, manifest at {output_dir / 'manifest.json'}")


@mood_map.command("analyze")
@click.option("--clips", required=True, help="Directory with clips + manifest.json")
@click.option("--output", required=True, help="Directory for PNG plots + report.txt")
@click.option(
    "--feature-sets", default="resemblyzer,spectral,combined,f0",
    help="Comma-separated feature sets to analyze",
)
@click.option("--tsne-perplexity", default=30.0, type=float, help="t-SNE perplexity (5-50)")
@click.option("--n-variance-components", default=10, type=int, help="PCA components for scree plot")
def mood_map_analyze(
    clips: str, output: str, feature_sets: str,
    tsne_perplexity: float, n_variance_components: int,
):
    """Analyze mood separation in acoustic feature spaces.

    Reads manifest.json from --clips, extracts features per clip,
    runs PCA and t-SNE, computes silhouette scores, writes PNG plots
    and a numeric report to --output.
    """
    from unkork.embeddings import get_encoder
    from unkork.mood_map import (
        FeatureAnalysis,
        analyze_feature_set,
        load_manifest,
        plot_explained_variance,
        plot_scatter,
        write_report,
    )

    clips_dir = Path(clips)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = clips_dir / "manifest.json"
    if not manifest_path.exists():
        raise click.ClickException(f"No manifest.json found in {clips_dir}")

    records = load_manifest(manifest_path)
    click.echo(f"Loaded {len(records)} clips from manifest")

    fs_list = [f.strip() for f in feature_sets.split(",")]

    # Load encoder once for feature sets that need it
    encoder = None
    if any(fs in ("resemblyzer", "combined") for fs in fs_list):
        click.echo("Loading Resemblyzer encoder...")
        encoder = get_encoder()

    analyses: list[FeatureAnalysis] = []
    for fs in fs_list:
        click.echo(f"\nAnalyzing: {fs}")

        def on_progress(done: int, total: int) -> None:
            if done % 50 == 0 or done == total:
                click.echo(f"  Extracting features: {done}/{total}")

        analysis = analyze_feature_set(
            records, fs,
            tsne_perplexity=tsne_perplexity,
            n_variance_components=n_variance_components,
            encoder=encoder,
            on_progress=on_progress,
        )
        analyses.append(analysis)

        click.echo(f"  Silhouette score: {analysis.silhouette:.4f}")

        plot_scatter(
            analysis.pca_2d, analysis.labels, analysis.voices,
            f"PCA — {fs} (silhouette={analysis.silhouette:.3f})",
            output_dir / f"pca_{fs}.png",
        )
        plot_scatter(
            analysis.tsne_2d, analysis.labels, analysis.voices,
            f"t-SNE — {fs} (silhouette={analysis.silhouette:.3f})",
            output_dir / f"tsne_{fs}.png",
        )
        plot_explained_variance(
            analysis.explained_variance, fs,
            output_dir / f"variance_{fs}.png",
        )
        click.echo(f"  Wrote plots to {output_dir}")

    report_path = output_dir / "report.txt"
    write_report(analyses, report_path)
    click.echo(f"\nReport: {report_path}")

    # Summary
    click.echo("\n" + "=" * 50)
    click.echo("RESULTS SUMMARY")
    click.echo("=" * 50)
    for a in analyses:
        click.echo(f"  {a.feature_set:<15} silhouette = {a.silhouette:+.4f}")
    click.echo("")
    best = max(analyses, key=lambda a: a.silhouette)
    click.echo(f"Best separation: {best.feature_set} ({best.silhouette:+.4f})")
