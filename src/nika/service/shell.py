"""Shared shell resolution and command wrapping for container exec."""

from __future__ import annotations

from typing import Protocol


SHELL_PROBE_CMD = (
    "/bin/sh -c 'if [ -x /bin/bash ]; then echo /bin/bash; "
    "elif [ -x /bin/sh ]; then echo /bin/sh; else echo /bin/sh; fi'"
)


def escape_for_shell_c(command: str) -> str:
    return command.replace("'", "'\\''").replace('"', '\\"')


def wrap_shell_command(shell: str, command: str) -> str:
    escaped = escape_for_shell_c(command)
    return f"{shell} -c '{escaped}'"


class ExecFn(Protocol):
    def __call__(self, node: str, cmd: str, *, timeout: float = 10.0) -> str: ...


class ShellResolver:
    """Cache per-node shell paths discovered via exec or lab metadata."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def resolve(
        self,
        node: str,
        exec_fn: ExecFn,
        *,
        preferred_shell: str | None = None,
    ) -> str:
        cached = self._cache.get(node)
        if cached is not None:
            return cached
        if preferred_shell is not None:
            self._cache[node] = preferred_shell
            return preferred_shell
        probed = exec_fn(node, SHELL_PROBE_CMD).strip()
        shell = probed if probed in ("/bin/bash", "/bin/sh") else "/bin/sh"
        self._cache[node] = shell
        return shell

    def exec_via_shell(
        self,
        node: str,
        command: str,
        exec_fn: ExecFn,
        *,
        preferred_shell: str | None = None,
        timeout: float = 10.0,
    ) -> str:
        shell = self.resolve(node, exec_fn, preferred_shell=preferred_shell)
        wrapped = wrap_shell_command(shell, command)
        return exec_fn(node, wrapped, timeout=timeout)
