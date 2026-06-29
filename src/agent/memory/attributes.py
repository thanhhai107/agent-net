"""Deterministic memory-attribute mining for network troubleshooting.

The LLM extractor may still propose attributes, but retrieval should not depend
only on model-generated labels. These helpers mine safe public attributes from
scenario/task/trace text without using benchmark oracle fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from agent.memory.models import MemoryAttributes

PROTOCOL_NAMES = (
    "bgp",
    "ospf",
    "rip",
    "dhcp",
    "dns",
    "arp",
    "icmp",
    "http",
    "https",
    "mpls",
    "p4",
    "bmv2",
    "vpn",
)

SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "dns": ("dns", "resolver", "resolv", "bind", "named"),
    "dhcp": ("dhcp", "lease", "leases", "dhclient", "dnsmasq", "kea"),
    "http": ("http", "https", "nginx", "apache", "web", "curl"),
    "ssh": ("ssh", "sshd"),
    "vpn": ("vpn", "ipsec", "wireguard", "openvpn"),
    "ntp": ("ntp", "chrony", "timesync"),
    "bgp": ("bgp", "bird"),
    "ospf": ("ospf",),
    "p4": ("p4", "bmv2", "simple_switch"),
}

SYMPTOM_PATTERNS: dict[str, tuple[str, ...]] = {
    "asymmetric reachability": (
        "asymmetric",
        "one-way",
        "one way",
        "only one direction",
    ),
    "packet loss": ("packet loss", "loss", "drops", "dropped"),
    "timeout": ("timeout", "timed out", "unresponsive"),
    "unreachable": ("unreachable", "no route", "destination host unreachable"),
    "route missing": ("missing route", "no route", "route absent", "blackhole"),
    "neighbor down": ("neighbor down", "session down", "adjacency down", "idle"),
    "service unavailable": (
        "service unavailable",
        "connection refused",
        "process down",
        "daemon down",
    ),
    "name resolution failure": (
        "dns failure",
        "resolution",
        "resolver",
        "name resolution",
        "cannot resolve",
    ),
    "lease failure": ("no lease", "lease failure", "dhcp failure", "dhcp timeout"),
    "latency": ("latency", "slow", "delay", "high rtt"),
}


def infer_memory_attributes(
    *texts: str,
    scenario: str = "",
    topology_class: str = "",
    task_stage: str = "diagnosis",
    tools: Iterable[str] = (),
) -> MemoryAttributes:
    """Infer safe, non-oracle attributes from public text and tool names."""

    haystack = " ".join(str(text or "") for text in (*texts, scenario)).lower()
    tokens = set(re.findall(r"[a-z0-9]+", haystack))
    tool_values = sorted({str(tool).strip().lower() for tool in tools if str(tool)})
    protocols = [
        name
        for name in PROTOCOL_NAMES
        if name in tokens or any(name in tool for tool in tool_values)
    ]
    services = [
        service
        for service, aliases in SERVICE_ALIASES.items()
        if any(alias in haystack for alias in aliases)
        or any(any(alias in tool for alias in aliases) for tool in tool_values)
    ]
    symptoms = [
        symptom
        for symptom, patterns in SYMPTOM_PATTERNS.items()
        if any(pattern in haystack for pattern in patterns)
    ]
    return MemoryAttributes(
        scenarios=[scenario] if scenario else [],
        topology_classes=[topology_class] if topology_class else [],
        protocols=protocols,
        services=services,
        task_stages=[task_stage] if task_stage else [],
        symptoms=symptoms,
        tools=tool_values,
    ).normalized()


def merge_memory_attributes(*attributes: MemoryAttributes) -> MemoryAttributes:
    """Union multiple MemoryAttributes values while preserving normalization."""

    merged: dict[str, list[str]] = {}
    for field_name in MemoryAttributes.model_fields:
        merged[field_name] = sorted(
            {
                value
                for attrs in attributes
                for value in getattr(attrs.normalized(), field_name)
            }
        )
    return MemoryAttributes(**merged).normalized()
