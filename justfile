# === Project: unkork ===
#
# Justfile conventions (see exocortex/scripts/justfile.template):
#
#   Universal:  develop, test, build, check, fmt, ship
#   Optional:   dev, clean
#
#   `ship` commits, pushes, and lets CI handle deployment.
#   `build` is the fast iterative check ("does the compiler love this?").
#   Deployment is a CI concern — see .forgejo/workflows/deploy.yml.

default:
    @just --list

# --- Environment ---

# Enter nix development shell
develop:
    nix develop

# Set up Python virtualenv via uv
setup:
    uv sync

# --- Quality ---

# Run tests
test:
    uv run pytest tests/ -v

# Format code
fmt:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

# Run all quality checks
check: fmt test

# --- Build ---

# Build the project (type check + lint)
build:
    uv run ruff check src/ tests/
    uv run pyright src/

# Remove build artifacts
clean:
    rm -rf .venv data/ models/ __pycache__ .pytest_cache .ruff_cache

# --- Shipping ---

# Commit and push. CI handles deployment.
ship message="ship":
    #!/usr/bin/env bash
    set -euo pipefail
    # Commit if tree is dirty
    if ! git diff --quiet HEAD 2>/dev/null \
        || ! git diff --cached --quiet 2>/dev/null \
        || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        git add -A
        git commit -m "{{message}}"
    fi
    git push
