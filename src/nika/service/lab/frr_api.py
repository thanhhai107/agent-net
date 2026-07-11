"""Shared FRR routing API for Kathara and Containerlab labs."""

from __future__ import annotations

import re

from nika.service.lab.protocols import SupportsExec


class FRRAPIMixin:
    """FRR routing daemon operations via ``exec_cmd``."""

    def frr_show_route(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip route'")

    def frr_exec(self: SupportsExec, device_name: str, command: str) -> str:
        return self.exec_cmd(device_name, f"vtysh -c '{command}'")

    def frr_show_running_config(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show running-config'")

    def frr_get_ospf_conf(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip ospf'")

    def frr_get_ospf_neighbors(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip ospf neighbor'")

    def frr_get_ospf_routes(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip route ospf'")

    def frr_get_ospf_interfaces(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip ospf interface'")

    def frr_get_bgp_conf(self: SupportsExec, device_name: str) -> str:
        return self.exec_cmd(device_name, "vtysh -c 'show ip bgp'")

    def frr_conf(self: SupportsExec, device_name: str, conf_commands: list[str]) -> str:
        command = 'vtysh -c "conf t"'
        for cmd in conf_commands:
            command += f' -c "{cmd}"'
        command += ' -c "end" -c "write"'
        return self.exec_cmd(device_name, command)

    def frr_add_route(
        self: SupportsExec, device_name: str, route: str, next_hop: str
    ) -> str:
        command = (
            f'vtysh -c "conf t" -c "ip route {route} {next_hop}" -c "end" -c "write"'
        )
        return self.exec_cmd(device_name, command)

    def frr_del_route(
        self: SupportsExec, device_name: str, route: str, next_hop: str
    ) -> str:
        command = (
            f'vtysh -c "conf t" -c "no ip route {route} {next_hop}" -c "end" -c "write"'
        )
        return self.exec_cmd(device_name, command)

    def frr_add_bgp_advertisement(
        self: SupportsExec, device_name: str, network: str, as_path: str
    ) -> str:
        command = (
            f'vtysh -c "conf t" -c "router bgp {as_path}" -c "network {network}" '
            f'-c "end" -c "write"'
        )
        return self.exec_cmd(device_name, command)

    def frr_del_bgp_advertisement(
        self: SupportsExec, device_name: str, network: str, as_path: str
    ) -> str:
        command = (
            f'vtysh -c "conf t" -c "router bgp {as_path}" -c "no network {network}" '
            f'-c "end" -c "write"'
        )
        return self.exec_cmd(device_name, command)

    def frr_get_bgp_asn_number(self: SupportsExec, node: str) -> int:
        summary = self.exec_cmd(
            node, "vtysh -c 'show bgp summary' 2>/dev/null || true"
        ).strip()
        match = re.search(r"local AS number\s+(\d+)", summary)
        if match:
            return int(match.group(1))

        running_config = self.exec_cmd(
            node,
            "vtysh -c 'show running-config' 2>/dev/null | grep -E '^router bgp ' | awk '{print $3}' | head -n1",
        ).strip()
        if running_config.isdigit():
            return int(running_config)

        raise ValueError(
            f"Could not determine BGP ASN for {node!r}. "
            f"summary={summary!r}, running_config_asn={running_config!r}"
        )
