"""Build and verify the agent sandbox Docker image."""

from __future__ import annotations

import shutil
import subprocess
import sys

from nika.config import _REPO_ROOT

SANDBOX_DOCKERFILE = _REPO_ROOT / "src" / "agent" / "sandbox" / "Dockerfile"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def sandbox_image_exists(image: str) -> bool:
    if not docker_available():
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def build_sandbox_image(
    image: str,
    *,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
) -> None:
    """Build the sandbox image from the repository Dockerfile."""
    if not docker_available():
        raise RuntimeError(
            "Docker is not available on PATH. Install Docker to use sandbox mode."
        )
    if not SANDBOX_DOCKERFILE.is_file():
        raise FileNotFoundError(f"Sandbox Dockerfile not found: {SANDBOX_DOCKERFILE}")

    cmd = [
        "docker",
        "build",
        "--network=host",
        "-t",
        image,
        "-f",
        str(SANDBOX_DOCKERFILE),
    ]
    if http_proxy:
        cmd.extend(["--build-arg", f"HTTP_PROXY={http_proxy}"])
        cmd.extend(["--build-arg", f"HTTPS_PROXY={https_proxy or http_proxy}"])
    cmd.append(str(_REPO_ROOT))

    print(f"Building sandbox image {image}...", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to build sandbox image {image}. "
            f"See docker build output above."
        )
    if not sandbox_image_exists(image):
        raise RuntimeError(f"Sandbox image {image} is missing after docker build")


def ensure_sandbox_image(
    image: str,
    *,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
) -> None:
    """Build the sandbox image when it is not present locally."""
    if sandbox_image_exists(image):
        return
    build_sandbox_image(
        image,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
    )
    print(f"Sandbox image ready: {image}", file=sys.stderr, flush=True)
