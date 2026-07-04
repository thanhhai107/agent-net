"""Kathara cleanup helpers used before experiment runs."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence


class KatharaCleanupError(RuntimeError):
    """Raised when the workspace cannot be made clean for a Kathara run."""


def _format_output(proc: subprocess.CompletedProcess[str]) -> str:
    chunks = []
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        chunks.append(f"stdout:\n{stdout}")
    if stderr:
        chunks.append(f"stderr:\n{stderr}")
    return "\n".join(chunks) if chunks else "(no output)"


def _run_checked(command: Sequence[str], *, step: str) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise KatharaCleanupError(
            f"Kathara cleanup failed during {step}: command not found: {command[0]!r}"
        ) from exc
    if proc.returncode != 0:
        raise KatharaCleanupError(
            f"Kathara cleanup failed during {step}: "
            f"{' '.join(command)} exited {proc.returncode}\n{_format_output(proc)}"
        )
    return proc


def _docker_lines(command: Sequence[str], *, step: str) -> list[str]:
    proc = _run_checked(command, step=step)
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def _kathara_container_ids(*, context: str) -> list[str]:
    ids: set[str] = set()
    filters = [
        ["--filter", "label=app=kathara"],
        ["--filter", "name=kathara_"],
    ]
    for extra_filter in filters:
        ids.update(
            _docker_lines(
                [
                    "docker",
                    "ps",
                    "-aq",
                    *extra_filter,
                ],
                step=f"{context}: docker container discovery",
            )
        )
    for row in _kathara_network_rows(context=context):
        network_id = row.split(maxsplit=1)[0]
        ids.update(_network_attached_container_ids(network_id, context=context))
    return sorted(ids)


def _remove_kathara_containers(*, context: str) -> None:
    container_ids = _kathara_container_ids(context=context)
    if not container_ids:
        return
    _run_checked(
        ["docker", "rm", "-f", *container_ids],
        step=f"{context}: docker container cleanup",
    )


def _kathara_network_rows(*, context: str) -> list[str]:
    rows_by_id: dict[str, str] = {}
    filters = [
        ["--filter", "label=app=kathara"],
        ["--filter", "name=kathara_"],
    ]
    for extra_filter in filters:
        for row in _docker_lines(
            [
                "docker",
                "network",
                "ls",
                *extra_filter,
                "--format",
                "{{.ID}} {{.Name}}",
            ],
            step=f"{context}: docker network verification",
        ):
            network_id = row.split(maxsplit=1)[0]
            rows_by_id[network_id] = row
    return [rows_by_id[network_id] for network_id in sorted(rows_by_id)]


def _network_containers(network_id: str, *, context: str) -> dict[str, object] | None:
    proc = _run_checked(
        [
            "docker",
            "network",
            "inspect",
            network_id,
            "--format",
            "{{json .Containers}}",
        ],
        step=f"{context}: docker network endpoint inspection",
    )
    try:
        containers = json.loads((proc.stdout or "{}").strip() or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(containers, dict):
        return None
    return containers


def _network_attached_container_ids(network_id: str, *, context: str) -> list[str]:
    containers = _network_containers(network_id, context=context)
    if containers is None:
        return []
    return sorted(containers)


def _network_has_visible_containers(network_id: str, *, context: str) -> bool:
    containers = _network_containers(network_id, context=context)
    return True if containers is None else bool(containers)


def _blocking_kathara_networks(*, context: str) -> list[str]:
    blocking: list[str] = []
    for row in _kathara_network_rows(context=context):
        network_id = row.split(maxsplit=1)[0]
        if _network_has_visible_containers(network_id, context=context):
            blocking.append(row)
    return blocking


def _remove_kathara_networks(*, context: str) -> None:
    rows = _kathara_network_rows(context=context)
    if not rows:
        return
    network_ids = [row.split(maxsplit=1)[0] for row in rows]
    _run_checked(
        ["docker", "network", "rm", *network_ids],
        step=f"{context}: docker network cleanup",
    )


def _retry_cleanup_step(
    command: Sequence[str],
    *,
    step: str,
    context: str,
    attempts: int = 3,
) -> subprocess.CompletedProcess[str]:
    last_error: KatharaCleanupError | None = None
    for attempt in range(attempts):
        try:
            return _run_checked(command, step=step)
        except KatharaCleanupError as exc:
            last_error = exc
            try:
                _remove_kathara_containers(context=context)
            except KatharaCleanupError:
                pass
            if "has active endpoints" in str(exc):
                try:
                    has_blocking_networks = bool(_blocking_kathara_networks(context=context))
                except KatharaCleanupError:
                    has_blocking_networks = True
                if not has_blocking_networks:
                    return subprocess.CompletedProcess(
                        list(command),
                        0,
                        stdout="Ignored stale Kathara networks with no visible containers.",
                        stderr="",
                    )
            if attempt + 1 < attempts:
                time.sleep(1.0)
    assert last_error is not None
    raise last_error


def ensure_kathara_clean(*, context: str = "run") -> None:
    """Wipe Kathara and fail if Docker still reports Kathara resources."""

    _retry_cleanup_step(
        ["kathara", "wipe", "-f"],
        step=f"{context}: kathara wipe",
        context=context,
    )
    _remove_kathara_containers(context=context)
    _retry_cleanup_step(
        ["docker", "network", "prune", "-f", "--filter", "label=app=kathara"],
        step=f"{context}: docker network prune",
        context=context,
    )
    for attempt in range(3):
        try:
            _remove_kathara_networks(context=context)
            break
        except KatharaCleanupError:
            _remove_kathara_containers(context=context)
            if attempt == 2:
                raise
            time.sleep(1.0)

    containers = _docker_lines(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "label=app=kathara",
            "--format",
            "{{.ID}} {{.Names}} {{.Status}}",
        ],
        step=f"{context}: docker container verification",
    )
    networks = _blocking_kathara_networks(context=context)
    if containers or networks:
        details = []
        if containers:
            details.append("containers:\n" + "\n".join(containers))
        if networks:
            details.append("networks:\n" + "\n".join(networks))
        raise KatharaCleanupError(
            f"Kathara cleanup incomplete before {context}.\n" + "\n".join(details)
        )
