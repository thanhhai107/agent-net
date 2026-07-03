"""Deterministic attribute mining for procedural skill retrieval."""

from __future__ import annotations

from agent.memory.models import MemoryAttributes


PROTOCOL_KEYWORDS = {
    "bgp": ("bgp", "asn", "route advertisement", "blackhole"),
    "ospf": ("ospf", "area", "neighbor", "lsa"),
    "dhcp": ("dhcp", "lease", "gateway", "subnet", "default route"),
    "dns": ("dns", "record", "resolve", "lookup", "nslookup", "dig", "servfail"),
    "http": ("http", "curl", "apache", "nginx", "load balancer", "web"),
    "icmp": ("icmp", "ping", "reachability"),
    "arp": ("arp", "neighbor cache", "mac address"),
    "p4": ("p4", "bmv2", "table", "pipeline"),
    "sdn": ("sdn", "controller", "openflow", "ovs"),
}

SERVICE_KEYWORDS = {
    "routing": ("route", "routing", "neighbor", "advertisement"),
    "name_resolution": ("dns", "resolve", "lookup"),
    "addressing": ("dhcp", "ip", "gateway", "netmask", "lease"),
    "web_service": ("http", "curl", "apache", "nginx", "load balancer", "web"),
    "forwarding": ("p4", "bmv2", "flow", "table"),
    "link_layer": ("arp", "mac", "interface", "carrier", "link"),
    "control_plane": ("controller", "frr", "bgp", "ospf", "openflow"),
}

SYMPTOM_KEYWORDS = {
    "unreachable": ("unreachable", "cannot reach", "ping fail", "blackhole"),
    "missing_route": ("missing route", "not advertised", "no route"),
    "service_down": ("down", "crash", "not running"),
    "latency_or_loss": ("latency", "loss", "delay", "corruption", "throttl"),
    "acl_block": ("acl", "blocked", "filter"),
    "missing_ip": ("missing ip", "no ipv4", "no ip address", "network is unreachable"),
    "bad_gateway": ("incorrect gateway", "wrong gateway", "default via", "default gateway"),
    "duplicate_ip": ("duplicate", "ip conflict", "address conflict"),
    "dns_failure": ("servfail", "nxdomain", "no such host", "resolver", "resolv.conf"),
    "link_down": ("link down", "state down", "no-carrier", "carrier down"),
    "link_flap": ("flap", "intermittent", "unstable link"),
    "table_miss": ("table entry missing", "table miss", "no entry"),
    "policy_violation": ("directly reachable", "should not be directly accessible"),
    "overload": ("dos", "contention", "resource contention", "incast"),
}


def _matches(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, needles in mapping.items() if any(item in text for item in needles)]


def _attribute_text(text: str) -> str:
    raw = str(text or "")
    observation_snippets = []
    for marker in (" -> ", "->"):
        if marker in raw:
            for part in raw.split(marker)[1:]:
                observation_snippets.append(part[:1600])
            break
    if observation_snippets:
        return " ".join(observation_snippets)
    if "Network Description:" in raw and "Your goal is" in raw:
        return raw.split("Your goal is", 1)[1]
    return raw


def infer_memory_attributes(
    text: str,
    *,
    scenario: str = "",
    topology_class: str = "",
    task_stage: str = "diagnosis",
    tools: list[str] | None = None,
) -> MemoryAttributes:
    # Scenario/topology names are stored separately and should not leak broad,
    # always-present design facts (for example OSPF/DHCP in every enterprise
    # task) into the transfer signature.
    haystack = " ".join([_attribute_text(text), " ".join(tools or [])]).lower()
    return MemoryAttributes(
        protocols=_matches(haystack, PROTOCOL_KEYWORDS),
        services=_matches(haystack, SERVICE_KEYWORDS),
        symptoms=_matches(haystack, SYMPTOM_KEYWORDS),
        task_stages=[task_stage] if task_stage else [],
        tools=list(tools or []),
    )
