"""Build and verify local Kathara Docker images shipped with NIKA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import docker
from docker.errors import BuildError, ImageNotFound

NIKA_IMAGE_PREFIX = "kathara/nika-"
DOCKER_FILES_DIR = Path(__file__).resolve().parent
NET_ENV_DIR = DOCKER_FILES_DIR.parents[1]

# Mirrors src/nika/net_env/utils/DockerFiles/build_dockers.sh plus local
# scenario-specific images that are not published on Docker Hub.
LOCAL_IMAGE_DOCKERFILES: dict[str, Path] = {
    "kathara/nika-frr": DOCKER_FILES_DIR / "Dockerfile.frr",
    "kathara/nika-base": DOCKER_FILES_DIR / "Dockerfile.base",
    "kathara/nika-nginx": DOCKER_FILES_DIR / "Dockerfile.nginx",
    "kathara/nika-wireguard": DOCKER_FILES_DIR / "Dockerfile.wireguard",
    "kathara/nika-pox": DOCKER_FILES_DIR / "Dockerfile.pox",
    "kathara/influxdb": NET_ENV_DIR / "p4" / "p4_int" / "Dockerfile",
}

_client: docker.DockerClient | None = None


def _get_client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def image_exists(image: str) -> bool:
    try:
        _get_client().images.get(image)
        return True
    except ImageNotFound:
        return False


def _dockerfile_for_image(image: str) -> Path:
    dockerfile = LOCAL_IMAGE_DOCKERFILES.get(image)
    if dockerfile is None:
        suffix = image.removeprefix(NIKA_IMAGE_PREFIX)
        dockerfile = DOCKER_FILES_DIR / f"Dockerfile.{suffix}"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"No Dockerfile for image {image}: {dockerfile}")
    return dockerfile


def build_nika_image(image: str) -> None:
    dockerfile = _dockerfile_for_image(image)
    print(f"Building Docker image {image} from {dockerfile.name}...")
    try:
        _, build_log = _get_client().images.build(
            path=str(dockerfile.parent),
            dockerfile=dockerfile.name,
            tag=image,
            rm=True,
        )
        for chunk in build_log:
            if "stream" in chunk:
                print(chunk["stream"], end="")
            elif "error" in chunk:
                raise BuildError(chunk["error"], build_log)
    except BuildError as exc:
        raise RuntimeError(f"Failed to build Docker image {image}") from exc


def ensure_nika_docker_images(required_images: Iterable[str]) -> None:
    """Build missing local images needed by a lab."""
    local_images = {
        img
        for img in required_images
        if img.startswith(NIKA_IMAGE_PREFIX) or img in LOCAL_IMAGE_DOCKERFILES
    }
    missing = {img for img in local_images if not image_exists(img)}
    if not missing:
        return

    print(f"Missing Docker images: {', '.join(sorted(missing))}")
    for image in sorted(missing):
        build_nika_image(image)

    still_missing: Set[str] = {img for img in missing if not image_exists(img)}
    if still_missing:
        raise RuntimeError("Failed to build required Docker images: " + ", ".join(sorted(still_missing)))
