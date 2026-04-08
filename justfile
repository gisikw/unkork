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

# --- Quality ---

# Run tests
test:
    @echo "TODO: define test command"
    # Go:     go test ./...
    # Elixir: mix test
    # Rust:   cargo test

# Format code
fmt:
    @echo "TODO: define fmt command"
    # Go:     go fmt ./...
    # Elixir: mix format
    # Rust:   cargo fmt
    # Nix:    nixfmt .

# Run all quality checks
check: fmt test

# --- Build ---

# Build the project (iterative/dev build — "does the compiler love this?")
build:
    @echo "TODO: define build command"
    # Go:     go build -o bin/unkork .
    # Elixir: mix compile --warnings-as-errors
    # Rust:   cargo build

# Remove build artifacts
# clean:
#     rm -rf result bin/

# --- Development ---

# Run in development mode (live reload, local server, etc.)
# dev:
#     # Go:     go run . --dev
#     # Elixir: iex -S mix phx.server
#     # Rust:   cargo tauri dev

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
