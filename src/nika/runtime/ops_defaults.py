"""Default exec-based implementations for LabRuntime semantic operations."""

from __future__ import annotations

import base64
import json
import shlex
from typing import Literal


def node_status_from_container(container) -> str:
    container.reload()
    return str(container.status)


def get_interface_operstate(runtime, node: str, intf: str) -> str:
    quoted = shlex.quote(intf)
    output = runtime.exec(node, f"cat /sys/class/net/{quoted}/operstate")
    return output.strip().lower()


def set_interface_state(runtime, node: str, intf: str, state: Literal["up", "down"]) -> str:
    quoted = shlex.quote(intf)
    return runtime.exec(node, f"ip link set {quoted} {state}")


def get_host_ip(runtime, node: str, iface: str = "eth0", *, with_prefix: bool = False) -> str | None:
    output = runtime.exec(node, "ip -j addr")
    try:
        ifaces = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse `ip -j addr` output on {node}: {exc}") from exc

    def format_ip(ip: str, prefix: int | None) -> str:
        if with_prefix and prefix is not None:
            return f"{ip}/{prefix}"
        return ip

    for link in ifaces:
        if link.get("ifname") != iface:
            continue
        for addr in link.get("addr_info", []):
            if addr.get("family") != "inet":
                continue
            ip = addr.get("local")
            prefix = addr.get("prefixlen")
            if ip and not ip.startswith("127."):
                return format_ip(ip, prefix)

    for link in ifaces:
        for addr in link.get("addr_info", []):
            if addr.get("family") != "inet":
                continue
            ip = addr.get("local")
            prefix = addr.get("prefixlen")
            if ip and not ip.startswith("127."):
                return format_ip(ip, prefix)
    return None


def get_default_gateway(runtime, node: str) -> str | None:
    output = runtime.exec(node, "ip -j route")
    try:
        routes = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse `ip -j route` output on {node}: {exc}") from exc
    for route in routes:
        if route.get("dst") == "default":
            gateway = route.get("gateway")
            if gateway:
                return gateway
    return None


def get_host_interfaces(runtime, node: str, *, include_loopback: bool = False) -> list[str]:
    output = runtime.exec(node, "ip -j addr")
    try:
        ifaces = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse `ip -j addr` output on {node}: {exc}") from exc
    names: list[str] = []
    for link in ifaces:
        name = link.get("ifname")
        if not name:
            continue
        if not include_loopback and name == "lo":
            continue
        if "br" in name:
            continue
        names.append(name)
    return names


def get_host_mac_address(runtime, node: str, iface: str = "eth0") -> str | None:
    quoted = shlex.quote(iface)
    result = runtime.exec(node, f"cat /sys/class/net/{quoted}/address").strip()
    return result or None


def list_nft_ruleset(runtime, node: str) -> str:
    return runtime.exec(node, "nft list ruleset 2>/dev/null").strip()


def _nft_add_chain(runtime, node: str, table: str, chain: str, family: str, hook: str) -> None:
    command = (
        f"nft add chain {family} {table} {chain} "
        f"'{{ type filter hook {hook} priority 0 ; policy accept ; }}'"
    )
    runtime.exec(node, command)


def add_nft_drop_rule(
    runtime,
    node: str,
    rule: str,
    *,
    table: str = "filter",
    family: str = "inet",
) -> None:
    runtime.exec(node, f"nft add table {family} {table}")
    for chain_name in ("input", "forward", "output"):
        _nft_add_chain(runtime, node, table, chain_name, family, chain_name)
        runtime.exec(node, f"nft add rule {family} {table} {chain_name} {rule}")


def delete_nft_table(runtime, node: str, *, table: str = "filter", family: str = "inet") -> None:
    runtime.exec(node, f"nft delete table {family} {table}")


def tc_set_netem(
    runtime,
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
    return runtime.exec(node, command)


def tc_set_tbf(
    runtime,
    node: str,
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
    return runtime.exec(node, command)


def tc_clear_intf(runtime, node: str, intf_name: str) -> str:
    return runtime.exec(node, f"tc qdisc del dev {intf_name} root")


def tc_show_intf(runtime, node: str, intf_name: str) -> str:
    return runtime.exec(node, f"tc qdisc show dev {intf_name}")


def systemctl(runtime, node: str, service: str, operation: Literal["start", "stop", "restart"]) -> str:
    return runtime.exec(node, f"systemctl {operation} {service}")


def kill_process(runtime, node: str, process_name: str) -> str:
    return runtime.exec(node, f"pkill -9 {process_name} 2>/dev/null; true")


def write_file(runtime, node: str, path: str, content: str) -> str:
    encoded = base64.b64encode(content.encode()).decode()
    quoted_path = shlex.quote(path)
    return runtime.exec(node, f"echo {encoded} | base64 -d > {quoted_path}")


def renew_dhcp_leases(runtime, nodes: list[str], intf: str = "eth0") -> None:
    quoted_intf = shlex.quote(intf)
    for node in nodes:
        runtime.exec(node, f"dhclient -r {quoted_intf}")
        runtime.exec(node, f"dhclient -v {quoted_intf}")


def _subnet_escaped(subnet: str) -> str:
    return subnet.replace(".", "\\.")


def dhcp_set_option_routers(runtime, dhcp_server: str, subnet: str, gateway: str) -> None:
    sub = _subnet_escaped(subnet)
    cmd = (
        f"sed -i '/subnet {sub} netmask 255\\.255\\.255\\.0/,/}}/ "
        f"s/option routers .*/option routers {gateway};/' /etc/dhcp/dhcpd.conf"
    )
    runtime.exec(dhcp_server, cmd)
    runtime.systemctl(dhcp_server, "isc-dhcp-server", "restart")


def dhcp_set_option_dns(runtime, dhcp_server: str, subnet: str, dns: str) -> None:
    sub = _subnet_escaped(subnet)
    cmd = (
        f"sed -i '/subnet {sub} netmask 255\\.255\\.255\\.0/,/}}/ "
        f"s/option domain-name-servers .*/option domain-name-servers {dns};/' /etc/dhcp/dhcpd.conf"
    )
    runtime.exec(dhcp_server, cmd)
    runtime.systemctl(dhcp_server, "isc-dhcp-server", "restart")


def dhcp_delete_subnet(runtime, dhcp_server: str, subnet: str) -> None:
    sub = _subnet_escaped(subnet)
    runtime.exec(dhcp_server, "cp /etc/dhcp/dhcpd.conf /etc/dhcp/dhcpd.conf.bak")
    runtime.exec(dhcp_server, f"sed -i '/subnet {sub} netmask 255\\.255\\.255\\.0/,/}}/d' /etc/dhcp/dhcpd.conf")
    runtime.systemctl(dhcp_server, "isc-dhcp-server", "restart")
