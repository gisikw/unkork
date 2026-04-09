# unkork

A regression codec for Kokoro TTS voice creation. Learns the inverse of Kokoro-82M's
unreleased style encoder by training a small neural network that maps audio features
to Kokoro voice tensors — enabling instant voice tensor generation from reference audio.

## How It Works

Kokoro-82M ships decoder-only — no encoder to create new voice tensors from audio.
unkork builds that missing encoder as a learned regression:

1. **Generate training data**: Create synthetic voices by blending existing Kokoro
   voice tensors, synthesize phrases for each, extract audio features
2. **Train**: Fit an MLP (audio features → PCA-compressed voice tensor space)
3. **Predict**: For any new target voice, extract features and run through
   the MLP to get a predicted Kokoro voice tensor instantly
4. **Refine** (optional): Polish the prediction with CMA-ES optimization against
   reference audio using a multi-axis scoring function

## Features

### Augmented Input Features (~476-dim)

The MLP takes a concatenation of:
- **Resemblyzer speaker embedding** (256-dim) — speaker identity
- **Librosa spectral features** (~220-dim) — MFCCs, spectral centroid/rolloff/bandwidth,
  spectral contrast, zero crossing rate, RMS energy, F0 (pyin), mel-spectrogram statistics

The spectral features capture tonal qualities (warmth, breathiness, resonance, register)
that Resemblyzer deliberately discards in favor of speaker identity.

### 4-Axis Refinement Scoring

The `refine` command uses CMA-ES optimization with a composite scoring function:

| Axis | Default Weight | What It Measures |
|------|---------------|-----------------|
| **mel** | 0.30 | Frame-level spectral shape (cosine distance) |
| **spectral** | 0.30 | Tonal characteristics — MFCCs, resonance, warmth |
| **f0** | 0.15 | Register/pitch match (weighted toward pitch center, not variability) |
| **resemblyzer** | 0.25 | Speaker identity anchor, prevents dialect drift |

Scores are combined via weighted harmonic mean — no single axis can be near-zero
without tanking the composite. All weights are tunable via CLI flags.

F0 is scored independently from other spectral features so refinement can target
register without flattening pitch dynamics or expressiveness.

## Usage

```bash
# Generate training dataset (synthesizes phrases, extracts features)
unkork generate --voices-dir /path/to/kokoro/voices --output data/ --count 1000

# Train the regression model
unkork train --data data/ --output models/ --epochs 500 --hidden-dim 1024

# Predict a voice tensor from reference audio
unkork predict --model models/codec.pt --pca models/pca.npz --audio reference.wav --output my_voice.pt

# Refine with 4-axis composite scoring (default weights)
unkork refine --start my_voice.pt --reference-dir refs/ --pca models/pca.npz \
  --iterations 50 --popsize 12

# Tune scoring weights (e.g., emphasize register match)
unkork refine --start my_voice.pt --reference-dir refs/ --pca models/pca.npz \
  --mel-weight 0.30 --spectral-weight 0.25 --f0-weight 0.25 --resemblyzer-weight 0.20

# Use individual scorers
unkork refine --start my_voice.pt --reference-dir refs/ --pca models/pca.npz --scorer f0
unkork refine --start my_voice.pt --reference-dir refs/ --pca models/pca.npz --scorer mel

# Quick audition
unkork speak --voice my_voice.pt --text "The morning light caught the copper."
```

### Iterative Refinement Loop

For overnight runs, chain refinements with audio snapshots:

```bash
cp my_voice.pt checkpoint_0.pt
i=1
while true; do
  unkork refine --start checkpoint_$((i-1)).pt --reference-dir refs/ \
    --pca models/pca.npz --iterations 50 --popsize 12 --output checkpoint_${i}.pt
  unkork speak --voice checkpoint_${i}.pt --output sample_${i}.wav
  echo "=== Checkpoint ${i} done ==="
  ((i++))
done
```

## Training Knobs

| Flag | Default | Notes |
|------|---------|-------|
| `--count` | 500 | Synthetic voices to generate. 1000+ recommended with augmented features |
| `--phrases` | 5 | Phrases per voice. 5 is stable; more gives diminishing returns |
| `--hidden-dim` | 512 | MLP hidden layer width. 1024 recommended for ~476-dim input |
| `--n-components` | 50 | PCA dimensions. 50 explains ~99.9% variance |
| `--epochs` | 200 | Training epochs. 500 with early stopping for larger models |

## Container Deployment

unkork runs in a podman container with CUDA support. The Nix flake produces a
wrapper script that lazily builds the container image on first invocation.

HuggingFace model weights cache to `/var/lib/unkork/cache/huggingface` (persistent
across runs). The spacy `en_core_web_sm` model is baked into the image at build time.

## Requirements

- Python 3.11+
- NVIDIA GPU recommended for training data generation (Kokoro synthesis)
- ~2GB disk for training data (more with higher `--count`)
