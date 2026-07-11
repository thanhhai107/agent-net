"""Startup verification signals for the rip_small_internet_vpn scenario."""

from __future__ import annotations

from typing import Any

from nika.net_env.verify import (
    build_lab_verify_result,
    exec_or_empty,
    host_has_ipv4,
    http_ok,
    nodes_deployed,
    ping_ok,
    service_active,
)
from nika.runtime.base import LabRuntime


def verify_rip_vpn_lab(runtime: LabRuntime, *, scenario_name: str) -> dict[str, Any]:
    expected = (
        "router1",
        "router2",
        "gateway_router",
        "external_router_1",
        "pc1",
        "pc2",
        "vpn_server_1",
        "web_server_1_1",
    )
    checks = {
        "nodes_deployed": nodes_deployed(runtime, expected),
        "router1_frr_active": service_active(runtime, "router1", "frr"),
        "pc1_ipv4": host_has_ipv4(runtime, "pc1", "10.0.0.2"),
        "pc1_gateway_reachable": ping_ok(runtime, "pc1", "10.0.0.1"),
        "pc1_to_pc2_reachable": ping_ok(runtime, "pc1", "10.0.1.2"),
        "external_web_reachable": ping_ok(runtime, "pc1", "20.0.0.3"),
        "wireguard_client": exec_or_empty(runtime, "pc1", "wg show wg0").strip() != "",
        "wireguard_server": exec_or_empty(
            runtime, "vpn_server_1", "wg show wg0"
        ).strip()
        != "",
        "web_service_active": service_active(runtime, "web_server_1_1", "apache2"),
        "web_http": http_ok(runtime, "pc1", "http://172.16.1.21/"),
    }
    return build_lab_verify_result(
        scenario_name=scenario_name,
        verified=all(checks.values()),
        checks=checks,
    )
