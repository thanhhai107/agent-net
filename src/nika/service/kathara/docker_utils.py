"""Resolve Docker containers for Kathara lab machines."""

from __future__ import annotations

from typing import Any

from docker.models.containers import Container
from Kathara.manager.Kathara import Kathara


def get_machine_container(*, lab_name: str, host_name: str) -> Container:
    """Return the Docker container for ``host_name`` inside ``lab_name``."""
    stats = next(
        Kathara.get_instance().get_machine_stats(machine_name=host_name, lab_name=lab_name),
        None,
    )
    if stats is None:
        raise ValueError(f"No container found for host {host_name!r} in lab {lab_name!r}.")
    return stats.machine_api_object


def list_lab_containers(*, lab_name: str) -> list[dict[str, Any]]:
    """Return running Kathara devices for ``lab_name`` (docker-ps-like metadata)."""
    containers = Kathara.get_instance().get_machines_api_objects(lab_name=lab_name)
    rows: list[dict[str, Any]] = []
    for container in containers:
        labels = container.labels or {}
        image = container.image.tags[0] if container.image.tags else container.image.short_id
        rows.append(
            {
                "container_id": container.short_id,
                "name": labels.get("name", "—"),
                "container_name": container.name.lstrip("/"),
                "image": image,
                "status": container.status,
            }
        )
    return sorted(rows, key=lambda row: row["name"])
