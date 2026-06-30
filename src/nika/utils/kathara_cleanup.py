"""Kathara cleanup helpers used before experiment runs."""

from __future__ import annotations

import subprocess
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


def ensure_kathara_clean(*, context: str = "run") -> None:
    """Wipe Kathara and fail if Docker still reports Kathara resources."""

    _run_checked(["kathara", "wipe", "-f"], step=f"{context}: kathara wipe")
    _run_checked(
        ["docker", "network", "prune", "-f", "--filter", "label=app=kathara"],
        step=f"{context}: docker network prune",
    )

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
    networks = _docker_lines(
        [
            "docker",
            "network",
            "ls",
            "--filter",
            "label=app=kathara",
            "--format",
            "{{.ID}} {{.Name}}",
        ],
        step=f"{context}: docker network verification",
    )
    if containers or networks:
        details = []
        if containers:
            details.append("containers:\n" + "\n".join(containers))
        if networks:
            details.append("networks:\n" + "\n".join(networks))
        raise KatharaCleanupError(
            f"Kathara cleanup incomplete before {context}.\n" + "\n".join(details)
        )
