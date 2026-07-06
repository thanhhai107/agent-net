"""LabRuntime protocol for Kathara and Containerlab backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from docker.models.containers import Container

from nika.runtime import ops_defaults


class LabRuntime(ABC):
    """Backend-neutral lab lifecycle, exec, and semantic observation APIs."""

    @property
    @abstractmethod
    def lab_name(self) -> str:
        """Logical lab name used by sessions and workflows."""

    @abstractmethod
    def deploy(self) -> None:
        """Deploy the lab if it is not already running."""

    @abstractmethod
    def destroy(self) -> None:
        """Tear down the lab."""

    @abstractmethod
    def exists(self) -> bool:
        """Return True when the lab has at least one running node."""

    @abstractmethod
    def inspect(self) -> list[dict[str, Any]]:
        """Return container rows aligned with ``list_lab_containers`` shape."""

    @abstractmethod
    def list_nodes(self) -> list[str]:
        """Return logical node names in the lab."""

    @abstractmethod
    def exec(self, node: str, cmd: str, *, timeout: float = 10.0) -> str:
        """Run a command inside ``node`` and return stdout/stderr text."""

    @abstractmethod
    def get_container(self, node: str) -> Container:
        """Return the Docker container for logical ``node``."""

    @abstractmethod
    def pause(self, node: str) -> None:
        """Pause the container backing ``node``."""

    @abstractmethod
    def unpause(self, node: str) -> None:
        """Unpause the container backing ``node``."""

    def node_status(self, node: str) -> str:
        """Return Docker container status for ``node`` (e.g. running, paused)."""
        try:
            return ops_defaults.node_status_from_container(self.get_container(node))
        except ValueError:
            return "not_found"

    def set_interface_state(self, node: str, intf: str, state: Literal["up", "down"]) -> str:
        return ops_defaults.set_interface_state(self, node, intf, state)

    def get_interface_operstate(self, node: str, intf: str) -> str:
        return ops_defaults.get_interface_operstate(self, node, intf)

    def get_host_ip(self, node: str, iface: str = "eth0", *, with_prefix: bool = False) -> str | None:
        return ops_defaults.get_host_ip(self, node, iface, with_prefix=with_prefix)

    def get_default_gateway(self, node: str) -> str | None:
        return ops_defaults.get_default_gateway(self, node)

    def get_host_interfaces(self, node: str, *, include_loopback: bool = False) -> list[str]:
        return ops_defaults.get_host_interfaces(self, node, include_loopback=include_loopback)

    def get_host_mac_address(self, node: str, iface: str = "eth0") -> str | None:
        return ops_defaults.get_host_mac_address(self, node, iface)

    def get_connected_devices(self, node: str) -> list[str]:
        """Return neighbor node names connected to ``node``; override when topology is available."""
        return []

    def list_nft_ruleset(self, node: str) -> str:
        return ops_defaults.list_nft_ruleset(self, node)

    def add_nft_drop_rule(
        self,
        node: str,
        rule: str,
        *,
        table: str = "filter",
        family: str = "inet",
    ) -> None:
        ops_defaults.add_nft_drop_rule(self, node, rule, table=table, family=family)

    def delete_nft_table(self, node: str, *, table: str = "filter", family: str = "inet") -> None:
        ops_defaults.delete_nft_table(self, node, table=table, family=family)

    def tc_set_netem(
        self,
        node: str,
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
        return ops_defaults.tc_set_netem(
            self,
            node,
            intf_name,
            loss=loss,
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            duplicate=duplicate,
            corrupt=corrupt,
            reorder=reorder,
            limit=limit,
            handle=handle,
            parent=parent,
        )

    def tc_set_tbf(
        self,
        node: str,
        intf_name: str,
        *,
        rate: str,
        burst: str,
        limit: str,
        handle: str | None = None,
        parent: str | None = None,
    ) -> str:
        return ops_defaults.tc_set_tbf(
            self,
            node,
            intf_name,
            rate=rate,
            burst=burst,
            limit=limit,
            handle=handle,
            parent=parent,
        )

    def tc_clear_intf(self, node: str, intf_name: str) -> str:
        return ops_defaults.tc_clear_intf(self, node, intf_name)

    def tc_show_intf(self, node: str, intf_name: str) -> str:
        return ops_defaults.tc_show_intf(self, node, intf_name)

    def systemctl(self, node: str, service: str, operation: Literal["start", "stop", "restart"]) -> str:
        return ops_defaults.systemctl(self, node, service, operation)

    def kill_process(self, node: str, process_name: str) -> str:
        return ops_defaults.kill_process(self, node, process_name)

    def write_file(self, node: str, path: str, content: str) -> str:
        return ops_defaults.write_file(self, node, path, content)

    def renew_dhcp_leases(self, nodes: list[str], intf: str = "eth0") -> None:
        ops_defaults.renew_dhcp_leases(self, nodes, intf)

    def dhcp_set_option_routers(self, dhcp_server: str, subnet: str, gateway: str) -> None:
        ops_defaults.dhcp_set_option_routers(self, dhcp_server, subnet, gateway)

    def dhcp_set_option_dns(self, dhcp_server: str, subnet: str, dns: str) -> None:
        ops_defaults.dhcp_set_option_dns(self, dhcp_server, subnet, dns)

    def dhcp_delete_subnet(self, dhcp_server: str, subnet: str) -> None:
        ops_defaults.dhcp_delete_subnet(self, dhcp_server, subnet)

    def list_dhcp_client_nodes(self) -> list[str]:
        return [node for node in self.list_nodes() if any(key in node for key in ("pc", "client"))]
