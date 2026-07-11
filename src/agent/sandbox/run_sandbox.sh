#!/usr/bin/env bash
# Run the NIKA agent sandbox with a restricted mount profile.
#
# Prefer production runs via: nika agent run --sandbox
# See docs/agent-sandbox.md for build instructions and configuration.
#
# - Only a temporary workspace is bind-mounted to /workspace
# - Runs as non-root user "agent"
# - No Docker socket, SSH keys, .env, or host home directory mounts
# - Network disabled by default (override with NIKA_SANDBOX_NETWORK=bridge)

set -euo pipefail

IMAGE="${NIKA_SANDBOX_IMAGE:-nika/agent-sandbox:latest}"
NETWORK="${NIKA_SANDBOX_NETWORK:-none}"

if [ "$#" -eq 0 ]; then
    set -- bash
fi

WORKSPACE="$(mktemp -d)"
cleanup() {
    rm -rf "$WORKSPACE"
}
trap cleanup EXIT

exec docker run --rm -it \
    --user agent \
    --network "$NETWORK" \
    --add-host=host.docker.internal:host-gateway \
    --security-opt no-new-privileges \
    -v "${WORKSPACE}:/workspace:rw" \
    "$IMAGE" \
    "$@"
