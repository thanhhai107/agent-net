"""Shared Docker container lifecycle helpers for lab runtimes."""

from __future__ import annotations

from docker.models.containers import Container


def pause_container(container: Container) -> None:
    container.reload()
    if container.status != "paused":
        container.pause()


def unpause_container(container: Container) -> None:
    container.reload()
    if container.status == "paused":
        container.unpause()
    elif container.status in {"created", "exited"}:
        container.start()
