"""Shared host interface API for Kathara and Containerlab labs."""

from __future__ import annotations

from typing import Literal

from nika.service.lab.protocols import SupportsExec


class IntfAPIMixin:
    """Host interface operations via ``exec_cmd``."""

    def intf_on_off(
        self: SupportsExec,
        host_name: str,
        interface: str,
        state: Literal["up", "down"],
    ) -> str:
        """Set a specific interface of a host on or off."""
        return self.exec_cmd(host_name, f"ip link set {interface} {state}")

    def intf_show(self: SupportsExec, host_name: str, interface: str) -> str:
        """Show the status of a specific interface of a host."""
        return self.exec_cmd(host_name, f"ip addr show {interface}")
