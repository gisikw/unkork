# Invariants

These are architectural contracts. They're not aspirational — they're load-bearing.
Violations are bugs, not style issues. No grandfather clauses.

If an invariant no longer serves the project, remove it explicitly with rationale.
Don't just ignore it.

## Code Organization

- Decision logic is pure: functions take data, return decisions, no I/O
- I/O is plumbing: thin orchestrators that gather data -> call pure functions -> act
- No multi-purpose functions: separate decision from effect
- New logic goes into testable functions first, not handler/framework layer

## File Size

- 500 lines max per file (ergonomic, not aesthetic)
- Split along behavioral seams, not alphabetically
- Tests mirror source files
- No grab-bag utility files (utils.py, helpers.py)

## Naming

- Timestamps: `2006-01-02_15-04-05` (lexicographic, filesystem-safe)

## Secrets

- No hardcoded secrets, tokens, PII, or infrastructure-specific details
- Environment-specific values come from .env (gitignored)
- .env.example documents required vars with placeholders

## Policy

- Decisions that shape code are explicit (here), not implicit
- No "look at how X does it" as policy — write it down or it doesn't exist

---

## Project-Specific

### What This Is

A regression codec that learns the inverse of Kokoro's missing style encoder.
Trains a small MLP mapping Resemblyzer speaker embeddings (256-dim) to Kokoro
voice tensors (511x1x256). Train once, infer instantly for any new voice.

### Architecture

- **CLI tool, not a service.** No overlay, no daemon. Runs interactively on GPU hosts.
- **Pipeline stages are separate commands:** generate training data, train, infer.
  Each stage reads from and writes to the filesystem. No in-memory coupling between stages.
- **Python with uv for dependency management.** System deps (ffmpeg, libsndfile) come
  from the Nix devShell. Python packages come from uv/pip.

### Data Model

- Training data lives in `data/` (gitignored): voice tensors, synthesized audio, embeddings
- Trained models live in `models/` (gitignored): PCA transforms, MLP checkpoints
- Voice tensors (input/output) are PyTorch `.pt` files, shape `511 x 1 x 256`
- Speaker embeddings are 256-dim float32 vectors (Resemblyzer)

### Dependencies

- **PyTorch** — MLP training and tensor I/O
- **Resemblyzer** — speaker embedding extraction
- **Kokoro/kokoro-onnx** — TTS synthesis for training data generation
- **cma** — optional CMA-ES refinement after MLP prediction
- **numpy, scipy** — PCA, numerical operations

### Testing

- Unit tests for pure functions (PCA transform, tensor reshaping, dataset construction)
- Integration tests can mock Kokoro synthesis with pre-recorded audio
- No tests that require GPU or model weights — those are manual validation
