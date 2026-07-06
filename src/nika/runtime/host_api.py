"""Host command API backed by LabRuntime (Containerlab sessions)."""

from __future__ import annotations

from typing import Literal

from func_timeout import FunctionTimedOut, func_timeout

from nika.runtime.base import LabRuntime
from nika.runtime.factory import resolve_backend, runtime_for_session
from nika.service.kathara.base_api import KatharaBaseAPI


class RuntimeHostAPI:
    """Minimal host exec API compatible with KatharaBaseAPI callers."""

    def __init__(self, runtime: LabRuntime) -> None:
        self.runtime = runtime
        self.lab_name = runtime.lab_name
        self._resolved_shell_cache: dict[str, str] = {}

    @staticmethod
    def _escape_for_shell_c(command: str) -> str:
        return command.replace("'", "'\\''").replace('"', '\\"')

    def _wrap_shell_command(self, shell: str, command: str) -> str:
        escaped = self._escape_for_shell_c(command)
        return f"{shell} -c '{escaped}'"

    def _resolve_shell(self, host_name: str) -> str:
        cached = self._resolved_shell_cache.get(host_name)
        if cached is not None:
            return cached
        probe_cmd = (
            "/bin/sh -c 'if [ -x /bin/bash ]; then echo /bin/bash; "
            "elif [ -x /bin/sh ]; then echo /bin/sh; else echo /bin/sh; fi'"
        )
        probed = self.runtime.exec(host_name, probe_cmd).strip()
        shell = probed if probed in ("/bin/bash", "/bin/sh") else "/bin/sh"
        self._resolved_shell_cache[host_name] = shell
        return shell

    def exec_cmd(self, host_name: str, command: str, timeout: float = 10) -> str:
        shell = self._resolve_shell(host_name)
        cmd = self._wrap_shell_command(shell, command)
        try:
            return func_timeout(timeout, self.runtime.exec, args=(host_name, cmd))
        except FunctionTimedOut:
            return f"[TIMEOUT] Command '{command}' on '{host_name}' exceeded {timeout}s."

    def intf_on_off(self, host_name: str, interface: str, state: Literal["up", "down"]) -> str:
        command = f"ip link set {interface} {state}"
        return self.exec_cmd(host_name, command)


def create_host_api(
    *,
    lab_name: str,
    backend: str = "kathara",
    runtime: LabRuntime | None = None,
    session_meta: dict | None = None,
):
    """Return KatharaBaseAPI or RuntimeHostAPI depending on backend."""
    if backend == "kathara":
        return KatharaBaseAPI(lab_name=lab_name)
    if runtime is None:
        if session_meta is None:
            session_meta = {"lab_name": lab_name, "backend": backend}
        runtime = runtime_for_session(session_meta)
    return RuntimeHostAPI(runtime)
