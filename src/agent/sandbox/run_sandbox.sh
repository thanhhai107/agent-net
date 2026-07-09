#!/usr/bin/env bash
# Run the NIKA agent sandbox with a restricted mount profile.
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
    -v "${WORKSPACE}:/workspace:rw" \
    "$IMAGE" \
    "$@"
