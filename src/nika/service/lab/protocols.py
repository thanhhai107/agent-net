"""Protocols shared by backend-neutral lab service APIs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsExec(Protocol):
    backend: str

    def exec_cmd(self, host_name: str, command: str, timeout: float = 10) -> str: ...
