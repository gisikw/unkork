{
  description = "unkork — regression codec for Kokoro TTS voice creation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "unkork";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          installPhase = ''
            mkdir -p $out/bin $out/share/unkork

            # Bundle source and Containerfile for lazy image build
            cp Containerfile $out/share/unkork/
            cp pyproject.toml $out/share/unkork/
            cp -r src $out/share/unkork/

            # Copy overlay definition
            cp overlay.nix $out/

            # Extract store hash for image tagging
            STORE_HASH=$(echo $out | grep -oP '/nix/store/\K[a-z0-9]+')

            # Create wrapper script
            cat > $out/bin/unkork <<WRAPPER
            #!/usr/bin/env bash
            set -euo pipefail

            IMAGE_TAG="unkork:$STORE_HASH"
            SHARE_DIR="$out/share/unkork"

            # Lazy build: materialize container on first run
            if ! podman image exists "\$IMAGE_TAG" 2>/dev/null; then
              echo "Building unkork container (first run)..." >&2
              podman build -t "\$IMAGE_TAG" -f "\$SHARE_DIR/Containerfile" "\$SHARE_DIR"
            fi

            exec podman run --rm -it \
              --device=nvidia.com/gpu=all \
              --shm-size=4g \
              -v /var/lib/unkork:/data:Z \
              -v /var/lib/fort/drops:/drops:ro \
              "\$IMAGE_TAG" "\$@"
            WRAPPER
            chmod +x $out/bin/unkork
          '';
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            just
            python312
            uv
            ffmpeg
            libsndfile
          ];

          shellHook = ''
            echo "unkork — just --list for recipes"
            export UV_PYTHON=${pkgs.python312}/bin/python3
            if [ ! -d .venv ]; then
              echo "Run 'uv sync' to set up the Python environment"
            fi
          '';

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.libsndfile
            pkgs.stdenv.cc.cc.lib
          ];
        };
      }
    );
}
