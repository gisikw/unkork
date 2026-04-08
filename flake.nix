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
