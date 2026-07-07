"""Resolve inject parameters when generating benchmark YAML (offline only)."""

from __future__ import annotations

from collections import defaultdict

from nika.net_env.net_env_pool import (
    get_net_env_instance,
    list_all_net_envs,
    scenario_backend,
)
from nika.problems.prob_pool import list_avail_problem_instances

_DEVICE_KEYS = ("host_name", "host_name_2", "attacker_device")


def _first(items: list[str] | None) -> str | None:
    return items[0] if items else None


def _second(items: list[str] | None) -> str | None:
    if items and len(items) > 1:
        return items[1]
    return _first(items)


def _pick(name: str | None, pool: list[str], fallback: str) -> str:
    if name and name in pool:
        return name
    return _first(pool) or fallback


def _dns_record_targets(net_env) -> tuple[str, str]:
    urls = getattr(net_env, "web_urls", None) or []
    if urls:
        url = urls[0]
        website = url.split(".")[0]
        if website.startswith("http://"):
            website = website[len("http://") :]
        domain = url.split(".")[1] if "." in url else "local"
        return website, domain
    web = _first(net_env.servers.get("web"))
    if web:
        return web.replace("web_server_", "web"), "local"
    return "web0", "local"


def _mac_conflict_pair(net_env) -> tuple[str, str]:
    topo = net_env.get_topology()
    if not topo:
        hosts = net_env.hosts
        return _first(hosts) or "pc1", _second(hosts) or "pc2"
    link = topo[0]
    device_a = link[0].split(":")[0]
    device_b = link[1].split(":")[0]
    return device_a, device_b


def _flow_rule_loop_pair(net_env) -> tuple[str, str]:
    switches = net_env.ovs_switches
    if len(switches) >= 2:
        return switches[0], switches[1]
    return _first(switches) or "leaf_1", _second(switches) or "leaf_2"


def _all_device_names(net_env) -> set[str]:
    names: set[str] = (
        set(net_env.lab.machines.keys()) if net_env.lab is not None else set()
    )
    names.update(net_env.hosts or [])
    names.update(net_env.routers or [])
    names.update(net_env.bmv2_switches or [])
    names.update(net_env.ovs_switches or [])
    names.update(net_env.sdn_controllers or [])
    for bucket in (net_env.servers or {}).values():
        names.update(bucket)
    names.update(getattr(net_env, "kubernetes_nodes", []) or [])
    if net_env.lab is None:
        spec = net_env.get_lab_spec()
        if spec is not None:
            names.update(node.name for node in spec.nodes)
    return names


def _get_net_env_for_benchmark(scenario: str, topo_size: str = ""):
    kwargs: dict = {}
    if topo_size:
        kwargs["topo_size"] = topo_size
    kwargs["backend"] = scenario_backend(scenario)
    return get_net_env_instance(scenario, **kwargs)


def _load_inventory(net_env) -> None:
    if net_env.lab is not None:
        net_env.load_machines()
        return

    spec = net_env.get_lab_spec()
    if spec is None:
        raise ValueError(f"Cannot derive benchmark inventory for {net_env.name!r}.")

    net_env.bmv2_switches = []
    net_env.ovs_switches = []
    net_env.sdn_controllers = []
    net_env.hosts = []
    net_env.routers = []
    net_env.switches = []
    net_env.servers = defaultdict(list)

    for node in spec.nodes:
        name = node.name
        kind = node.kind.lower()
        image = node.image.lower()
        if any(key in name for key in ("client", "pc", "host")) or "linux" in kind:
            net_env.hosts.append(name)
        elif any(key in kind for key in ("srl", "ceos", "router")) or any(
            key in image for key in ("srl", "ceos", "frr")
        ):
            net_env.routers.append(name)
        else:
            net_env.switches.append(name)

    net_env.hosts = sorted(net_env.hosts)
    net_env.routers = sorted(net_env.routers)
    net_env.switches = sorted(net_env.switches)


def _scenario_device_defaults(scenario: str, net_env) -> dict[str, str]:
    """Preferred host/router targets for k8s and llmd labs."""
    hosts = net_env.hosts or []
    routers = net_env.routers or []
    k8s_nodes = getattr(net_env, "kubernetes_nodes", []) or []

    if scenario == "k8s_lab":
        return {
            "host0": _pick("client", hosts, "client"),
            "host1": _pick("client", hosts, "client"),
            "router0": _pick("leaf_1_1", routers, _first(routers) or "leaf_1_1"),
            "web0": _pick("client", hosts, "client"),
            "attacker": _pick("client", hosts, "client"),
        }
    if scenario == "llmd_lab":
        client = _pick("client", hosts, "client")
        controller = _pick("controller", k8s_nodes, _first(k8s_nodes) or client)
        return {
            "host0": client,
            "host1": client,
            "router0": controller,
            "web0": client,
            "attacker": client,
            "controller": controller,
        }
    if scenario == "min3clos":
        return {
            "host0": _pick("client1", hosts, _first(hosts) or "client1"),
            "host1": _pick("client2", hosts, _second(hosts) or "client2"),
            "router0": _pick("leaf1", routers, _first(routers) or "leaf1"),
            "web0": _pick("client1", hosts, "client1"),
            "attacker": _pick("client2", hosts, "client2"),
        }
    return {}


def resolve_inject_params(
    problem: str, scenario: str, topo_size: str = ""
) -> dict[str, str]:
    """Return inject params for one benchmark row."""
    net_env = _get_net_env_for_benchmark(scenario, topo_size)
    _load_inventory(net_env)

    hosts = net_env.hosts
    routers = net_env.routers
    servers = net_env.servers
    bmv2 = net_env.bmv2_switches
    controllers = net_env.sdn_controllers

    defaults = _scenario_device_defaults(scenario, net_env)
    host0 = defaults.get("host0") or _first(hosts) or "pc1"
    host1 = defaults.get("host1") or _second(hosts) or host0
    router0 = defaults.get("router0") or _first(routers) or host0
    dns0 = _first(servers.get("dns")) or host0
    dhcp0 = _first(servers.get("dhcp")) or dns0
    web0 = defaults.get("web0") or _first(servers.get("web")) or host0
    vpn0 = _first(servers.get("vpn")) or host0
    lb0 = _first(servers.get("load_balancer")) or web0
    attacker = defaults.get("attacker") or (hosts[-1] if hosts else host0)

    params: dict[str, str] = {}

    if problem in {
        "link_down",
        "link_flap",
        "link_detach",
        "link_fragmentation_disabled",
        "link_high_packet_corruption",
        "link_bandwidth_throttling",
        "host_missing_ip",
        "host_incorrect_ip",
        "host_incorrect_gateway",
        "host_incorrect_netmask",
        "host_incorrect_dns",
        "host_crash",
        "arp_cache_poisoning",
        "receiver_resource_contention",
    }:
        if scenario == "min3clos" and problem.startswith("link_"):
            params["host_name"] = router0
            params["intf_name"] = "e1-1"
        else:
            params["host_name"] = host0
            if problem.startswith("link_"):
                params["intf_name"] = "eth0"
        if problem == "link_flap":
            params["down_time"] = "30"
            params["up_time"] = "30"
        if problem == "link_fragmentation_disabled":
            params["mtu"] = "100"
        if problem == "host_incorrect_netmask":
            params["netmask_prefix"] = "8"
        if problem == "link_bandwidth_throttling":
            params["rate"] = "30kbit"
            params["burst"] = "64kb"
            params["limit"] = "500kb"
        if problem == "link_high_packet_corruption":
            params["corruption_percentage"] = "60"
        if problem == "receiver_resource_contention":
            params["duration"] = "600"

    elif problem == "host_ip_conflict":
        params["host_name"] = host0
        params["host_name_2"] = host1

    elif problem == "dns_record_error":
        website, domain = _dns_record_targets(net_env)
        params["host_name"] = dns0
        params["target_website"] = website
        params["target_domain"] = domain

    elif problem in {"dns_service_down"}:
        params["host_name"] = dns0

    elif problem in {"dhcp_service_down", "dhcp_missing_subnet"}:
        params["host_name"] = dhcp0
        params["host_name_2"] = host0

    elif problem in {"dhcp_spoofed_gateway", "dhcp_spoofed_dns", "dhcp_spoofed_subnet"}:
        params["host_name"] = dhcp0
        params["host_name_2"] = host0

    elif problem == "host_vpn_membership_missing":
        params["host_name"] = host0
        params["host_name_2"] = vpn0

    elif problem in {
        "bgp_acl_block",
        "bgp_asn_misconfig",
        "bgp_missing_route_advertisement",
        "host_static_blackhole",
        "bgp_blackhole_route_leak",
        "bgp_hijacking",
        "ospf_acl_block",
        "ospf_area_misconfiguration",
        "ospf_neighbor_missing",
        "frr_service_down",
    }:
        params["host_name"] = router0

    elif problem in {"arp_acl_block", "icmp_acl_block", "http_acl_block"}:
        params["host_name"] = host0

    elif problem == "dns_port_blocked":
        params["host_name"] = dns0

    elif problem == "mac_address_conflict":
        a, b = _mac_conflict_pair(net_env)
        params["host_name"] = a
        params["host_name_2"] = b

    elif problem in {
        "p4_header_definition_error",
        "p4_compilation_error_parser_state",
        "p4_table_entry_missing",
        "p4_table_entry_misconfig",
        "p4_aggressive_detection_thresholds",
        "bmv2_switch_down",
        "mpls_label_limit_exceeded",
    }:
        params["host_name"] = _first(bmv2) or host0

    elif problem in {
        "sdn_controller_crash",
        "southbound_port_block",
        "southbound_port_mismatch",
    }:
        params["host_name"] = _first(controllers) or host0
        if problem == "southbound_port_block":
            params["southbound_port"] = "6633"
        if problem == "southbound_port_mismatch":
            params["mismatched_port"] = "6653"
            params["original_port"] = "6633"

    elif problem == "flow_rule_shadowing":
        params["host_name"] = _first(net_env.ovs_switches) or host0

    elif problem == "flow_rule_loop":
        a, b = _flow_rule_loop_pair(net_env)
        params["host_name"] = a
        params["host_name_2"] = b

    elif problem == "web_dos_attack":
        if scenario == "llmd_lab":
            params["host_name"] = defaults.get("controller") or host0
            params["attacker_device"] = attacker
        else:
            params["host_name"] = web0
            params["attacker_device"] = attacker

    elif problem == "dns_lookup_latency":
        params["host_name"] = dns0 if dns0 in _all_device_names(net_env) else host0
        params["intf_name"] = "eth0"
        params["delay_ms"] = "1000"

    elif problem == "incast_traffic_network_limitation":
        params["host_name"] = web0 if web0 in (servers.get("web") or []) else host0
        params["rate"] = "1mbit"
        params["burst"] = "500kb"
        params["limit"] = "500kb"
        params["delay_ms"] = "20"

    elif problem in {"sender_resource_contention", "sender_application_delay"}:
        params["host_name"] = web0 if web0 in (servers.get("web") or []) else host0
        if problem == "sender_resource_contention":
            params["duration"] = "600"

    elif problem == "load_balancer_overload":
        params["host_name"] = lb0
        params["duration"] = "300"

    else:
        params["host_name"] = host0

    return params


def validate_benchmark_case(
    scenario: str,
    problem: str,
    inject: dict[str, str],
    topo_size: str = "",
) -> None:
    """Raise ValueError if a benchmark row is inconsistent with tags or topology."""
    net_envs = list_all_net_envs()
    problems = list_avail_problem_instances()
    if scenario not in net_envs:
        raise ValueError(f"Unknown scenario {scenario!r}")
    if problem not in problems:
        raise ValueError(f"Unknown problem {problem!r}")

    problem_tags = set(problems[problem].TAGS)
    scenario_tags = set(net_envs[scenario].TAGS)
    if not problem_tags.issubset(scenario_tags):
        raise ValueError(
            f"Tag mismatch for {problem} on {scenario}: "
            f"problem tags {sorted(problem_tags)} not subset of scenario tags {sorted(scenario_tags)}"
        )

    net_env = _get_net_env_for_benchmark(scenario, topo_size)
    _load_inventory(net_env)
    devices = _all_device_names(net_env)

    for key in _DEVICE_KEYS:
        value = inject.get(key)
        if value and value not in devices:
            raise ValueError(
                f"Inject device {key}={value!r} not in {scenario} topology "
                f"(topo_size={topo_size!r}); known devices: {sorted(devices)}"
            )

    if problem == "dns_record_error":
        website = inject.get("target_website", "")
        domain = inject.get("target_domain", "")
        urls = getattr(net_env, "web_urls", None) or []
        if urls:
            matched = any(
                website in url and (not domain or domain in url) for url in urls
            )
            if not matched:
                raise ValueError(
                    f"DNS record targets {website}.{domain} not found in web_urls for {scenario}: {urls}"
                )
