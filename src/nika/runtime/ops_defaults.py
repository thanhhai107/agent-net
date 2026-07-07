"""Default exec-based implementations for LabRuntime semantic operations."""

from __future__ import annotations

import base64
import json
import re
import shlex
from typing import Literal


def node_status_from_container(container) -> str:
    container.reload()
    return str(container.status)


def get_interface_operstate(runtime, node: str, intf: str) -> str:
    quoted = shlex.quote(intf)
    output = runtime.exec(node, f"cat /sys/class/net/{quoted}/operstate")
    return output.strip().lower()


def set_interface_state(
    runtime, node: str, intf: str, state: Literal["up", "down"]
) -> str:
    quoted = shlex.quote(intf)
    return runtime.exec(node, f"ip link set {quoted} {state}")


def get_host_ip(
    runtime, node: str, iface: str = "eth0", *, with_prefix: bool = False
) -> str | None:
    output = runtime.exec(node, "ip -j addr")
    try:
        ifaces = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse `ip -j addr` output on {node}: {exc}"
        ) from exc

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
        raise RuntimeError(
            f"Failed to parse `ip -j route` output on {node}: {exc}"
        ) from exc
    for route in routes:
        if route.get("dst") == "default":
            gateway = route.get("gateway")
            if gateway:
                return gateway
    return None


def get_host_interfaces(
    runtime, node: str, *, include_loopback: bool = False
) -> list[str]:
    output = runtime.exec(node, "ip -j addr")
    try:
        ifaces = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse `ip -j addr` output on {node}: {exc}"
        ) from exc
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


def _nft_add_chain(
    runtime, node: str, table: str, chain: str, family: str, hook: str
) -> None:
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


def delete_nft_table(
    runtime, node: str, *, table: str = "filter", family: str = "inet"
) -> None:
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


def systemctl(
    runtime, node: str, service: str, operation: Literal["start", "stop", "restart"]
) -> str:
    return runtime.exec(node, f"systemctl {operation} {service}")


def frr_get_bgp_asn_number(runtime, node: str) -> int:
    summary = runtime.exec(
        node, "vtysh -c 'show bgp summary' 2>/dev/null || true"
    ).strip()
    match = re.search(r"local AS number\s+(\d+)", summary)
    if match:
        return int(match.group(1))

    running_config = runtime.exec(
        node,
        "vtysh -c 'show running-config' 2>/dev/null | grep -E '^router bgp ' | awk '{print $3}' | head -n1",
    ).strip()
    if running_config.isdigit():
        return int(running_config)

    raise ValueError(
        f"Could not determine BGP ASN for {node!r}. "
        f"summary={summary!r}, running_config_asn={running_config!r}"
    )


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


def dhcp_set_option_routers(
    runtime, dhcp_server: str, subnet: str, gateway: str
) -> None:
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
    runtime.exec(
        dhcp_server,
        f"sed -i '/subnet {sub} netmask 255\\.255\\.255\\.0/,/}}/d' /etc/dhcp/dhcpd.conf",
    )
    runtime.systemctl(dhcp_server, "isc-dhcp-server", "restart")


def process_running(runtime, node: str, process_name: str) -> bool:
    quoted = shlex.quote(process_name)
    output = runtime.exec(node, f"pgrep -a {quoted} 2>/dev/null || echo NONE").strip()
    return output != "NONE" and process_name in output


def process_not_running(runtime, node: str, process_name: str) -> bool:
    return not process_running(runtime, node, process_name)


def pidfile_running(runtime, node: str, pidfile: str) -> bool:
    quoted = shlex.quote(pidfile)
    output = runtime.exec(
        node,
        f"if [ -f {quoted} ] && kill -0 $(cat {quoted}) 2>/dev/null; then echo running; else echo not_running; fi",
    )
    return output.strip() == "running"


def interface_exists(runtime, node: str, intf: str) -> bool:
    quoted = shlex.quote(intf)
    output = runtime.exec(node, f"ip link show {quoted} 2>&1")
    lowered = output.lower()
    return "does not exist" not in lowered and "no such device" not in lowered


def tc_qdisc_contains(runtime, node: str, intf: str, keyword: str) -> bool:
    output = tc_show_intf(runtime, node, intf)
    return keyword.lower() in output.lower()


def iptables_rule_present(runtime, node: str, chain: str, rule_args: str) -> bool:
    check_cmd = f"iptables -C {chain} {rule_args} >/dev/null 2>&1 && echo present || echo absent"
    return runtime.exec(node, check_cmd).strip() == "present"


def nft_ruleset_contains(runtime, node: str, pattern: str) -> bool:
    return pattern in list_nft_ruleset(runtime, node)


def ping_ok(runtime, node: str, target: str, *, count: int = 1) -> bool:
    quoted = shlex.quote(target)
    output = runtime.exec(node, f"ping -c {count} -W 2 {quoted} 2>&1")
    return " 0% packet loss" in output or " 0% loss" in output


def dig_query(runtime, node: str, domain: str, *, nameserver: str | None = None) -> str:
    ns = f"@{shlex.quote(nameserver)} " if nameserver else ""
    quoted_domain = shlex.quote(domain)
    return runtime.exec(node, f"dig +short {ns}{quoted_domain} 2>/dev/null").strip()


def file_contains(runtime, node: str, path: str, pattern: str) -> bool:
    quoted_path = shlex.quote(path)
    quoted_pat = shlex.quote(pattern)
    output = runtime.exec(
        node,
        f"grep -q -F {quoted_pat} {quoted_path} 2>/dev/null && echo yes || echo no",
    )
    return output.strip() == "yes"


def start_background_od_traffic(
    runtime,
    od_dicts: dict[str, dict[str, int]],
    *,
    interval: int = 5,
    unit: str = "M",
    udp: bool = True,
    server_args: str = "",
    client_args: str = "",
) -> list[str]:
    """Start iperf3 OD-matrix traffic in the background via ``runtime.exec``."""
    started_server_ports: dict[str, int] = {}
    server_port_assign: dict[str, dict[str, int]] = {}
    labels: list[str] = []
    start_port_id = 5201

    for src_host, dests in od_dicts.items():
        for dst_host in dests:
            if src_host == dst_host:
                continue
            labels.append(f"{src_host}_to_{dst_host}")
            if dst_host in started_server_ports:
                started_server_ports[dst_host] += 1
                dst_port = started_server_ports[dst_host]
                server_port_assign[dst_host][src_host] = dst_port
            else:
                dst_port = start_port_id
                started_server_ports[dst_host] = start_port_id
                server_port_assign[dst_host] = {src_host: start_port_id}
            server_cmd = f"iperf3 -s -p {dst_port} {server_args} -J &"
            runtime.exec(dst_host, server_cmd)

    for src_host, dests in od_dicts.items():
        for dst_host, volume in dests.items():
            if src_host == dst_host:
                continue
            dst_ip = get_host_ip(runtime, dst_host)
            if not dst_ip:
                raise ValueError(f"Cannot resolve IP for host {dst_host!r}")
            dst_port = server_port_assign[dst_host][src_host]
            client_udp = " -u" if udp else ""
            client_cmd = (
                f"iperf3 -c {dst_ip} -p {dst_port} -b {volume}{unit} "
                f"-t {interval}{client_udp} {client_args} -l 1472 -J &"
            )
            runtime.exec(src_host, client_cmd)

    return labels
