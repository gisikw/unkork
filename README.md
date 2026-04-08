# unkork

A regression codec for Kokoro TTS voice creation. Learns the inverse of Kokoro-82M's
unreleased style encoder by training a small neural network that maps speaker embeddings
to Kokoro voice tensors.

## How It Works

Kokoro-82M ships decoder-only — no encoder to create new voice tensors from audio.
unkork builds that missing encoder as a learned regression:

1. **Generate training data**: Create ~500 synthetic voices by blending existing Kokoro
   voice tensors, synthesize phrases for each, compute speaker embeddings
2. **Train**: Fit an MLP (256-dim embedding → PCA-compressed voice tensor space)
3. **Infer**: For any new target voice, compute its speaker embedding and run through
   the MLP to get a predicted Kokoro voice tensor instantly
4. **Refine** (optional): Polish the prediction with a short CMA-ES optimization

## Usage

```bash
# Enter development environment
just develop

# Generate training dataset (~500 synthetic voices)
unkork generate --voices-dir /path/to/kokoro/voices --output data/

# Train the regression model
unkork train --data data/ --output models/

# Predict a voice tensor from reference audio
unkork predict --model models/codec.pt --audio reference.wav --output my_voice.pt

# Optional: refine with CMA-ES
unkork refine --start my_voice.pt --target reference.wav --iterations 50
```

## Requirements

- Python 3.11+
- GPU recommended for training data generation (Kokoro synthesis)
- ~2GB disk for training data
