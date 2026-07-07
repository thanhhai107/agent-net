"""Host command API backed by LabRuntime (Containerlab sessions)."""

from __future__ import annotations

from typing import Literal

from nika.runtime.base import LabRuntime
from nika.runtime.factory import runtime_for_session
from nika.runtime.shell import ShellResolver
from nika.service.kathara.base_api import KatharaBaseAPI


class RuntimeHostAPI:
    """Minimal host exec API compatible with KatharaBaseAPI callers."""

    def __init__(self, runtime: LabRuntime) -> None:
        self.runtime = runtime
        self.lab_name = runtime.lab_name
        self._shell = ShellResolver()

    def exec_cmd(self, host_name: str, command: str, timeout: float = 10) -> str:
        return self._shell.exec_via_shell(
            host_name,
            command,
            self.runtime.exec,
            timeout=timeout,
        )

    def intf_on_off(
        self, host_name: str, interface: str, state: Literal["up", "down"]
    ) -> str:
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
