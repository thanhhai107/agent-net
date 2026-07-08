"""Startup verification signals for the p4_int scenario."""

from __future__ import annotations

from typing import Any

from nika.net_env.kathara.p4.verify_utils import p4_switches_ready
from nika.net_env.verify import (
    build_lab_verify_result,
    host_has_ipv4,
    nodes_deployed,
    ping_ok,
    process_running,
)
from nika.runtime.base import LabRuntime


def verify_p4_int_lab(runtime: LabRuntime, *, scenario_name: str) -> dict[str, Any]:
    checks = {
        "nodes_deployed": nodes_deployed(
            runtime, ("pc1", "pc2", "collector", "leaf1", "leaf2", "spine1", "spine2")
        ),
        "p4_switches_ready": p4_switches_ready(
            runtime, ("leaf1", "leaf2", "spine1", "spine2")
        ),
        "collector_process": process_running(runtime, "collector", "python3"),
        "pc1_ipv4": host_has_ipv4(runtime, "pc1", "10.0.0.1"),
        "pc2_ipv4": host_has_ipv4(runtime, "pc2", "10.0.0.2"),
        "collector_ipv4": host_has_ipv4(runtime, "collector", "10.0.0.3"),
        "pc1_to_pc2_reachable": ping_ok(runtime, "pc1", "10.0.0.2"),
    }
    return build_lab_verify_result(
        scenario_name=scenario_name,
        verified=all(checks.values()),
        checks=checks,
    )
