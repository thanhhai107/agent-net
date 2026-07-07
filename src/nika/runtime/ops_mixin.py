"""Exec-based semantic operations mixed into LabRuntime backends."""

from __future__ import annotations

from typing import Literal

from nika.runtime import ops_defaults


class ExecSemanticOpsMixin:
    """Default semantic lab operations implemented via ``runtime.exec``."""

    require_capabilities: object
    exec: object
    get_container: object
    list_nodes: object

    def node_status(self, node: str) -> str:
        self.require_capabilities("node_status")
        try:
            return ops_defaults.node_status_from_container(self.get_container(node))
        except ValueError:
            return "not_found"

    def set_interface_state(
        self, node: str, intf: str, state: Literal["up", "down"]
    ) -> str:
        self.require_capabilities("interface")
        return ops_defaults.set_interface_state(self, node, intf, state)

    def get_interface_operstate(self, node: str, intf: str) -> str:
        self.require_capabilities("interface")
        return ops_defaults.get_interface_operstate(self, node, intf)

    def get_host_ip(
        self, node: str, iface: str = "eth0", *, with_prefix: bool = False
    ) -> str | None:
        self.require_capabilities("ip")
        return ops_defaults.get_host_ip(self, node, iface, with_prefix=with_prefix)

    def get_default_gateway(self, node: str) -> str | None:
        self.require_capabilities("route")
        return ops_defaults.get_default_gateway(self, node)

    def get_host_interfaces(
        self, node: str, *, include_loopback: bool = False
    ) -> list[str]:
        self.require_capabilities("interface")
        return ops_defaults.get_host_interfaces(
            self, node, include_loopback=include_loopback
        )

    def get_host_mac_address(self, node: str, iface: str = "eth0") -> str | None:
        self.require_capabilities("interface")
        return ops_defaults.get_host_mac_address(self, node, iface)

    def list_nft_ruleset(self, node: str) -> str:
        self.require_capabilities("nft")
        return ops_defaults.list_nft_ruleset(self, node)

    def add_nft_drop_rule(
        self,
        node: str,
        rule: str,
        *,
        table: str = "filter",
        family: str = "inet",
    ) -> None:
        self.require_capabilities("nft")
        ops_defaults.add_nft_drop_rule(self, node, rule, table=table, family=family)

    def delete_nft_table(
        self, node: str, *, table: str = "filter", family: str = "inet"
    ) -> None:
        self.require_capabilities("nft")
        ops_defaults.delete_nft_table(self, node, table=table, family=family)

    def tc_set_netem(
        self,
        node: str | None = None,
        intf_name: str = "",
        *,
        host_name: str | None = None,
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
        self.require_capabilities("tc")
        target_node = node if node is not None else host_name
        if target_node is None:
            raise TypeError("tc_set_netem requires node or host_name.")
        return ops_defaults.tc_set_netem(
            self,
            target_node,
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
        node: str | None = None,
        intf_name: str = "",
        *,
        host_name: str | None = None,
        rate: str,
        burst: str,
        limit: str,
        handle: str | None = None,
        parent: str | None = None,
    ) -> str:
        self.require_capabilities("tc")
        target_node = node if node is not None else host_name
        if target_node is None:
            raise TypeError("tc_set_tbf requires node or host_name.")
        return ops_defaults.tc_set_tbf(
            self,
            target_node,
            intf_name,
            rate=rate,
            burst=burst,
            limit=limit,
            handle=handle,
            parent=parent,
        )

    def tc_clear_intf(self, node: str, intf_name: str) -> str:
        self.require_capabilities("tc")
        return ops_defaults.tc_clear_intf(self, node, intf_name)

    def tc_show_intf(self, node: str, intf_name: str) -> str:
        self.require_capabilities("tc")
        return ops_defaults.tc_show_intf(self, node, intf_name)

    def systemctl(
        self, node: str, service: str, operation: Literal["start", "stop", "restart"]
    ) -> str:
        self.require_capabilities("service")
        return ops_defaults.systemctl(self, node, service, operation)

    def frr_get_bgp_asn_number(self, node: str) -> int:
        self.require_capabilities("frr")
        return ops_defaults.frr_get_bgp_asn_number(self, node)

    def kill_process(self, node: str, process_name: str) -> str:
        self.require_capabilities("process")
        return ops_defaults.kill_process(self, node, process_name)

    def write_file(self, node: str, path: str, content: str) -> str:
        self.require_capabilities("file")
        return ops_defaults.write_file(self, node, path, content)

    def renew_dhcp_leases(self, nodes: list[str], intf: str = "eth0") -> None:
        self.require_capabilities("dns")
        ops_defaults.renew_dhcp_leases(self, nodes, intf)

    def dhcp_set_option_routers(
        self, dhcp_server: str, subnet: str, gateway: str
    ) -> None:
        self.require_capabilities("dns")
        ops_defaults.dhcp_set_option_routers(self, dhcp_server, subnet, gateway)

    def dhcp_set_option_dns(self, dhcp_server: str, subnet: str, dns: str) -> None:
        self.require_capabilities("dns")
        ops_defaults.dhcp_set_option_dns(self, dhcp_server, subnet, dns)

    def dhcp_delete_subnet(self, dhcp_server: str, subnet: str) -> None:
        self.require_capabilities("dns")
        ops_defaults.dhcp_delete_subnet(self, dhcp_server, subnet)

    def list_dhcp_client_nodes(self) -> list[str]:
        self.require_capabilities("dns")
        return [
            node
            for node in self.list_nodes()
            if any(key in node for key in ("pc", "client"))
        ]

    def process_running(self, node: str, process_name: str) -> bool:
        self.require_capabilities("process")
        return ops_defaults.process_running(self, node, process_name)

    def process_not_running(self, node: str, process_name: str) -> bool:
        self.require_capabilities("process")
        return ops_defaults.process_not_running(self, node, process_name)

    def pidfile_running(self, node: str, pidfile: str) -> bool:
        self.require_capabilities("pidfile")
        return ops_defaults.pidfile_running(self, node, pidfile)

    def interface_exists(self, node: str, intf: str) -> bool:
        self.require_capabilities("interface")
        return ops_defaults.interface_exists(self, node, intf)

    def tc_qdisc_contains(self, node: str, intf: str, keyword: str) -> bool:
        self.require_capabilities("tc")
        return ops_defaults.tc_qdisc_contains(self, node, intf, keyword)

    def iptables_rule_present(self, node: str, chain: str, rule_args: str) -> bool:
        self.require_capabilities("iptables")
        return ops_defaults.iptables_rule_present(self, node, chain, rule_args)

    def nft_ruleset_contains(self, node: str, pattern: str) -> bool:
        self.require_capabilities("nft")
        return ops_defaults.nft_ruleset_contains(self, node, pattern)

    def ping_ok(self, node: str, target: str, *, count: int = 1) -> bool:
        self.require_capabilities("exec")
        return ops_defaults.ping_ok(self, node, target, count=count)

    def dig_query(
        self, node: str, domain: str, *, nameserver: str | None = None
    ) -> str:
        self.require_capabilities("dns")
        return ops_defaults.dig_query(self, node, domain, nameserver=nameserver)

    def file_contains(self, node: str, path: str, pattern: str) -> bool:
        self.require_capabilities("file")
        return ops_defaults.file_contains(self, node, path, pattern)

    def start_background_od_traffic(
        self,
        od_dicts: dict[str, dict[str, int]],
        *,
        interval: int = 5,
        unit: str = "M",
        udp: bool = True,
        server_args: str = "",
        client_args: str = "",
    ) -> list[str]:
        self.require_capabilities("traffic", "ip")
        return ops_defaults.start_background_od_traffic(
            self,
            od_dicts,
            interval=interval,
            unit=unit,
            udp=udp,
            server_args=server_args,
            client_args=client_args,
        )
