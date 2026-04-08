# unkork overlay — CLI-only, no service.
# Wrapper script lazily builds a podman container on first run,
# then forwards all args into the GPU-enabled container.
{ storePath, ... }:
{
  bins = [ "${storePath}/bin/unkork" ];
  health = { type = "none"; };
}
