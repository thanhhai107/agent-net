"""Deterministic attribute mining for procedural skill retrieval."""

from __future__ import annotations

from agent.memory.models import MemoryAttributes


PROTOCOL_KEYWORDS = {
    "bgp": ("bgp", "asn", "route advertisement", "blackhole"),
    "ospf": ("ospf", "area", "neighbor", "lsa"),
    "dhcp": ("dhcp", "lease", "gateway", "subnet"),
    "dns": ("dns", "record", "resolve", "lookup"),
    "icmp": ("icmp", "ping", "reachability"),
    "p4": ("p4", "bmv2", "table", "pipeline"),
}

SERVICE_KEYWORDS = {
    "routing": ("route", "routing", "neighbor", "advertisement"),
    "name_resolution": ("dns", "resolve", "lookup"),
    "addressing": ("dhcp", "ip", "gateway", "netmask"),
    "forwarding": ("p4", "bmv2", "flow", "table"),
}

SYMPTOM_KEYWORDS = {
    "unreachable": ("unreachable", "cannot reach", "ping fail", "blackhole"),
    "missing_route": ("missing route", "not advertised", "no route"),
    "service_down": ("down", "crash", "not running"),
    "latency_or_loss": ("latency", "loss", "delay", "corruption"),
    "acl_block": ("acl", "blocked", "filter"),
}


def _matches(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, needles in mapping.items() if any(item in text for item in needles)]


def infer_memory_attributes(
    text: str,
    *,
    scenario: str = "",
    topology_class: str = "",
    task_stage: str = "diagnosis",
    tools: list[str] | None = None,
) -> MemoryAttributes:
    haystack = " ".join([text, scenario, topology_class, " ".join(tools or [])]).lower()
    return MemoryAttributes(
        protocols=_matches(haystack, PROTOCOL_KEYWORDS),
        services=_matches(haystack, SERVICE_KEYWORDS),
        symptoms=_matches(haystack, SYMPTOM_KEYWORDS),
        task_stages=[task_stage] if task_stage else [],
        tools=list(tools or []),
    )
