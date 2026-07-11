"""Shared traffic-control API for Kathara and Containerlab labs."""

from __future__ import annotations

from nika.service.lab.protocols import SupportsExec


class TCMixin:
    """Linux ``tc`` operations via ``exec_cmd``."""

    def tc_set_netem(
        self: SupportsExec,
        host_name: str,
        intf_name: str,
        *,
        loss: int | None = None,
        delay_ms: int | None = None,
        jitter_ms: int | None = None,
        duplicate: int | None = None,
        corrupt: int | None = None,
        reorder: int | None = None,
        limit: int | None = None,
        handle: str | None = None,
        parent: str | None = None,
    ) -> str:
        command = f"tc qdisc add dev {intf_name}"
        if parent is not None:
            command += f" parent {parent}"
        else:
            command += " root"
        if handle is not None:
            handle = handle if handle.endswith(":") else handle + ":"
            command += f" handle {handle}"
        command += " netem"
        if loss is not None:
            command += f" loss {loss}%"
        if delay_ms is not None and jitter_ms is None:
            command += f" delay {delay_ms}ms"
        elif delay_ms is not None and jitter_ms is not None:
            command += f" delay {delay_ms}ms {jitter_ms}ms"
        if duplicate is not None:
            command += f" duplicate {duplicate}%"
        if reorder is not None:
            command += f" reorder {reorder}%"
        if corrupt is not None:
            command += f" corrupt {corrupt}%"
        if limit is not None:
            command += f" limit {limit}"
        return self.exec_cmd(host_name, command)

    def tc_set_tbf(
        self: SupportsExec,
        host_name: str,
        intf_name: str,
        *,
        rate: str,
        burst: str,
        limit: str,
        handle: str | None = None,
        parent: str | None = None,
    ) -> str:
        command = f"tc qdisc add dev {intf_name}"
        if parent is not None:
            command += f" parent {parent}"
        else:
            command += " root"
        if handle is not None:
            handle = handle if handle.endswith(":") else handle + ":"
            command += f" handle {handle}"
        command += f" tbf rate {rate} burst {burst} limit {limit}"
        return self.exec_cmd(host_name, command)

    def tc_clear_intf(self: SupportsExec, host_name: str, intf_name: str) -> str:
        return self.exec_cmd(host_name, f"tc qdisc del dev {intf_name} root")

    def tc_show_intf(self: SupportsExec, host_name: str, intf_name: str) -> str:
        return self.exec_cmd(host_name, f"tc qdisc show dev {intf_name}")

    def tc_show_statistics(self: SupportsExec, host_name: str, intf_name: str) -> str:
        return self.exec_cmd(host_name, f"tc -s qdisc show dev {intf_name}")

    def tc_qdisc_contains(
        self: SupportsExec, host_name: str, intf: str, keyword: str
    ) -> bool:
        output = self.tc_show_intf(host_name, intf)
        return keyword.lower() in output.lower()
