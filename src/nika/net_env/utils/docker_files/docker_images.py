"""Build and verify local NIKA Kathara Docker images via the Docker Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import docker
from docker.errors import BuildError, ImageNotFound

NIKA_IMAGE_PREFIX = "kathara/nika-"
DOCKER_FILES_DIR = Path(__file__).resolve().parent

# Mirrors src/nika/net_env/utils/DockerFiles/build_dockers.sh
NIKA_IMAGE_DOCKERFILES: dict[str, str] = {
    "kathara/nika-frr": "Dockerfile.frr",
    "kathara/nika-base": "Dockerfile.base",
    "kathara/nika-nginx": "Dockerfile.nginx",
    "kathara/nika-wireguard": "Dockerfile.wireguard",
    "kathara/nika-pox": "Dockerfile.pox",
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
    dockerfile_name = NIKA_IMAGE_DOCKERFILES.get(image)
    if dockerfile_name is None:
        suffix = image.removeprefix(NIKA_IMAGE_PREFIX)
        dockerfile_name = f"Dockerfile.{suffix}"
    dockerfile = DOCKER_FILES_DIR / dockerfile_name
    if not dockerfile.is_file():
        raise FileNotFoundError(f"No Dockerfile for image {image}: {dockerfile}")
    return dockerfile


def build_nika_image(image: str) -> None:
    dockerfile = _dockerfile_for_image(image)
    print(f"Building Docker image {image} from {dockerfile.name}...")
    try:
        _, build_log = _get_client().images.build(
            path=str(DOCKER_FILES_DIR),
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


def ensure_nika_docker_images(required_images: Iterable[str], *, force_rebuild: bool = False) -> None:
    """Build kathara/nika-* images needed by a lab.

    By default only missing images are built. With ``force_rebuild=True``, every
    required NIKA image is rebuilt even if it already exists locally.
    """
    nika_images = {img for img in required_images if img.startswith(NIKA_IMAGE_PREFIX)}
    if force_rebuild:
        to_build = nika_images
    else:
        to_build = {img for img in nika_images if not image_exists(img)}
    if not to_build:
        return

    if force_rebuild:
        print(f"Force rebuilding Docker images: {', '.join(sorted(to_build))}")
    else:
        print(f"Missing Docker images: {', '.join(sorted(to_build))}")
    for image in sorted(to_build):
        build_nika_image(image)

    still_missing: Set[str] = {img for img in to_build if not image_exists(img)}
    if still_missing:
        raise RuntimeError("Failed to build required Docker images: " + ", ".join(sorted(still_missing)))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build NIKA Kathara Docker images.")
    parser.add_argument(
        "-f",
        "--force-rebuild",
        action="store_true",
        help="Rebuild images even if they already exist locally.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="Images to build (default: all known kathara/nika-* images).",
    )
    args = parser.parse_args()
    required = args.images or list(NIKA_IMAGE_DOCKERFILES.keys())
    ensure_nika_docker_images(required, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
