"""Fault-family evidence gate for diagnosis workflows.

The gate is intentionally domain-level rather than benchmark-answer-level.  It
checks whether a final report has at least one current, discriminating tool
observation for the broad fault family it claims to diagnose.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


INTEGRATED_GUIDANCE_MARKER = "[Integrated learning guidance - not evidence]"
ENV_EVIDENCE_GATE_ENABLED = "NIKA_EVIDENCE_GATE_ENABLED"


def evidence_gate_enabled(default: bool = True) -> bool:
    """Return whether runtime evidence gating is enabled."""
    raw_value = os.getenv(ENV_EVIDENCE_GATE_ENABLED)
    if raw_value is None or not raw_value.strip():
        return bool(default)
    return raw_value.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }


@dataclass(frozen=True)
class ToolObservation:
    """One current-run tool observation used by the evidence gate."""

    tool: str = ""
    tool_input: str = ""
    summary: str = ""


@dataclass(frozen=True)
class EvidenceProbe:
    """A class of evidence that can support one fault family."""

    label: str
    description: str
    tool_patterns: tuple[str, ...] = ()
    text_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FaultFamily:
    """Broad NIKA/networking fault family, not a ground-truth answer."""

    key: str
    label: str
    trigger_patterns: tuple[str, ...]
    probes: tuple[EvidenceProbe, ...]


@dataclass(frozen=True)
class EvidenceGateResult:
    """Evidence-gate verdict and remediation prompt."""

    sufficient: bool
    families: tuple[str, ...]
    observed_tools: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    suggested_steps: tuple[str, ...]
    prompt: str = ""

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "families": list(self.families),
            "observed_tools": list(self.observed_tools),
            "missing_evidence": list(self.missing_evidence),
            "suggested_steps": list(self.suggested_steps),
            "prompt": self.prompt,
        }


FAMILY_REQUIREMENTS: tuple[FaultFamily, ...] = (
    FaultFamily(
        key="dns",
        label="DNS/resolver",
        trigger_patterns=(
            r"\bdns\b",
            r"\bresolv(?:e|er|ing|ution)\b",
            r"\bnameserver\b",
            r"\bdomain\b",
            r"\.local\b",
        ),
        probes=(
            EvidenceProbe(
                label="DNS resolution/config/service evidence",
                description=(
                    "direct DNS resolution, resolver configuration, DNS service, "
                    "or DNS port evidence"
                ),
                tool_patterns=(
                    "curl_web_test",
                    "cat_file",
                    "systemctl_ops",
                    "netstat",
                    "exec_shell",
                ),
                text_patterns=(
                    r"\bnslookup\b",
                    r"\bdig\b",
                    r"\bgetent\s+hosts\b",
                    r"\bresolv\.conf\b",
                    r"\bnameserver\b",
                    r"\bname_lookup\b",
                    r"\bnamelookup\b",
                    r"\bdnsmasq\b",
                    r"\bbind9?\b",
                    r"\bport\s+53\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="dhcp_host_config",
        label="DHCP/host IP configuration",
        trigger_patterns=(
            r"\bdhcp\b",
            r"\blease\b",
            r"\bgateway\b",
            r"\bdefault\s+route\b",
            r"\bincorrect\s+ip\b",
            r"\bmissing\s+ip\b",
            r"\bip\s+conf(?:ig|lict)\b",
            r"\bhost\s+(?:mis)?config",
        ),
        probes=(
            EvidenceProbe(
                label="host address, route, lease, or gateway evidence",
                description=(
                    "current host IP address, DHCP lease, default route, gateway, "
                    "or duplicate-address evidence"
                ),
                tool_patterns=(
                    "get_host_net_config",
                    "ip_addr_statistics",
                    "exec_shell",
                    "cat_file",
                ),
                text_patterns=(
                    r"\bip\s+addr\b",
                    r"\binet\s+\d+\.\d+\.\d+\.\d+",
                    r"\bdefault\s+via\b",
                    r"\bgateway\b",
                    r"\bdhcp\b",
                    r"\blease\b",
                    r"\bduplicate\b",
                    r"\bconflict\b",
                    r"\bresolv\.conf\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="physical_link",
        label="physical/interface link",
        trigger_patterns=(
            r"\blink\s+(?:down|flap)",
            r"\binterface\s+down\b",
            r"\bcarrier\b",
            r"\beth\d+\s+down\b",
            r"\bstate\s+down\b",
            r"\bflapping\b",
        ),
        probes=(
            EvidenceProbe(
                label="interface state/counter evidence",
                description=(
                    "interface state, carrier status, ethtool output, or interface "
                    "statistics from the suspected device"
                ),
                tool_patterns=(
                    "ethtool",
                    "ip_addr_statistics",
                    "get_host_net_config",
                    "exec_shell",
                ),
                text_patterns=(
                    r"\blink\s+detected:\s*no",
                    r"\bstate\s+down\b",
                    r"\bcarrier\b",
                    r"\bno-carrier\b",
                    r"\bflap",
                    r"\binterface\s+\S+\s+down\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="link_performance",
        label="link performance/impairment",
        trigger_patterns=(
            r"\bbandwidth\b",
            r"\bthrottl",
            r"\bpacket\s+(?:loss|corruption|corrupt)",
            r"\bcorruption\b",
            r"\blatency\b",
            r"\bthroughput\b",
            r"\biperf\b",
            r"\btc\b",
            r"\bnetem\b",
            r"\btbf\b",
        ),
        probes=(
            EvidenceProbe(
                label="traffic-control or throughput evidence",
                description=(
                    "tc/qdisc statistics, iperf throughput, curl timing, or "
                    "measured packet-loss evidence beyond broad reachability"
                ),
                tool_patterns=(
                    "get_tc_statistics",
                    "iperf_test",
                    "curl_web_test",
                    "exec_shell",
                ),
                text_patterns=(
                    r"\bqdisc\b",
                    r"\bnetem\b",
                    r"\btbf\b",
                    r"\brate\s+\d+",
                    r"\biperf\b",
                    r"\bMbits/sec\b",
                    r"\bKbits/sec\b",
                    r"\bpacket\s+loss\b",
                    r"\bcorruption\b",
                    r"\btime_total\b",
                    r"\bconnect_time\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="bgp",
        label="BGP control plane",
        trigger_patterns=(
            r"\bbgp\b",
            r"\basn\b",
            r"\bas-path\b",
            r"\badvertis",
            r"\bprefix\b",
            r"\bleaf_router\b",
            r"\bspine_router\b",
            r"\broute\s+leak\b",
            r"\bblackhole\b",
        ),
        probes=(
            EvidenceProbe(
                label="BGP neighbor/route/config evidence",
                description=(
                    "BGP neighbor state, advertised prefixes, BGP config, or route "
                    "table evidence from the relevant routers"
                ),
                tool_patterns=(
                    "frr_show_bgp_summary",
                    "frr_get_bgp_conf",
                    "frr_show_ip_route",
                    "frr_exec",
                ),
                text_patterns=(
                    r"\bbgp\b",
                    r"\bneighbor\b",
                    r"\bAS\b",
                    r"\brouter\s+bgp\b",
                    r"\badvertis",
                    r"\bprefix\b",
                    r"\bshow\s+ip\s+bgp\b",
                    r"\bblackhole\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="ospf_frr",
        label="OSPF/FRR routing",
        trigger_patterns=(
            r"\bospf\b",
            r"\bfrr\b",
            r"\brouter_core\b",
            r"\barea\s+\d+",
            r"\broute\s+missing\b",
            r"\brouting\s+service\b",
        ),
        probes=(
            EvidenceProbe(
                label="OSPF/FRR service, neighbor, config, or route evidence",
                description=(
                    "OSPF/FRR service state, neighbor state, OSPF config, or "
                    "routing table evidence"
                ),
                tool_patterns=(
                    "frr_get_ospf_conf",
                    "frr_show_ip_route",
                    "frr_exec",
                    "systemctl_ops",
                ),
                text_patterns=(
                    r"\bospf\b",
                    r"\bfrr\b",
                    r"\bdaemon\b",
                    r"\bneighbor\b",
                    r"\bshow\s+ip\s+ospf\b",
                    r"\bshow\s+ip\s+route\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="http_acl_service",
        label="HTTP/service/ACL",
        trigger_patterns=(
            r"\bhttp\b",
            r"\bweb\b",
            r"\bcurl\b",
            r"\bnginx\b",
            r"\bapache\b",
            r"\bservice\s+down\b",
            r"\bport\s+80\b",
            r"\bacl\b",
            r"\bfirewall\b",
            r"\bblocked\b",
            r"\bdos\b",
        ),
        probes=(
            EvidenceProbe(
                label="application/service/listener/filter evidence",
                description=(
                    "HTTP timing/status, service state, listener state, or ACL/drop "
                    "rule evidence"
                ),
                tool_patterns=(
                    "curl_web_test",
                    "systemctl_ops",
                    "netstat",
                    "exec_shell",
                    "cat_file",
                ),
                text_patterns=(
                    r"\bhttp\b",
                    r"\bHTTP/\d",
                    r"\bstatus\b",
                    r"\blisten\b",
                    r"\bport\s+80\b",
                    r"\bnginx\b",
                    r"\bapache\b",
                    r"\biptables\b",
                    r"\bnft\b",
                    r"\bdrop\b",
                    r"\breject\b",
                    r"\bSYN\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="arp_l2",
        label="ARP/L2 neighbor",
        trigger_patterns=(
            r"\barp\b",
            r"\bmac\b",
            r"\blladdr\b",
            r"\bneighbor\s+cache\b",
            r"\bpoison",
        ),
        probes=(
            EvidenceProbe(
                label="ARP/MAC neighbor evidence",
                description=(
                    "current ARP/neigh table, MAC mapping, or duplicate L2 "
                    "neighbor evidence"
                ),
                tool_patterns=(
                    "exec_shell",
                    "ip_addr_statistics",
                    "get_host_net_config",
                ),
                text_patterns=(
                    r"\barp\b",
                    r"\bip\s+neigh\b",
                    r"\blladdr\b",
                    r"\bmac\b",
                    r"\bFAILED\b",
                    r"\bSTALE\b",
                    r"\bREACHABLE\b",
                ),
            ),
        ),
    ),
    FaultFamily(
        key="p4_bmv2",
        label="P4/BMv2 data plane",
        trigger_patterns=(
            r"\bp4\b",
            r"\bbmv2\b",
            r"\btable\s+entry\b",
            r"\bregister\b",
            r"\bcounter\b",
            r"\bdata\s+plane\b",
        ),
        probes=(
            EvidenceProbe(
                label="P4 table/counter/register evidence",
                description=(
                    "BMv2 log, counter, table, register, or P4 program evidence"
                ),
                tool_patterns=(
                    "bmv2_get_log",
                    "bmv2_get_counter_arrays",
                    "bmv2_counter_read",
                    "bmv2_show_tables",
                    "bmv2_table_dump",
                    "bmv2_get_register_arrays",
                    "bmv2_register_read",
                    "bmv2_read_p4_program",
                ),
                text_patterns=(
                    r"\bbmv2\b",
                    r"\bp4\b",
                    r"\btable\b",
                    r"\bcounter\b",
                    r"\bregister\b",
                    r"\bpacket\b",
                ),
            ),
        ),
    ),
)


FINAL_CLAIM_RE = re.compile(
    r"\b(anomaly|faulty|fault|root cause|root_cause|caused by|"
    r"misconfig|down|blocked|incorrect|missing|poison|blackhole)\b",
    re.I,
)


def infer_fault_families(*texts: str) -> tuple[FaultFamily, ...]:
    """Infer broad fault families from visible task/report text."""

    haystack = "\n".join(str(text or "") for text in texts).lower()
    families: list[FaultFamily] = []
    for family in FAMILY_REQUIREMENTS:
        if any(re.search(pattern, haystack, re.I) for pattern in family.trigger_patterns):
            families.append(family)
    return tuple(families)


def observations_from_runtime_snapshot(snapshot: dict[str, Any] | None) -> list[ToolObservation]:
    """Build observations from Skill-Pro runtime state."""

    if not snapshot:
        return []
    observations: list[ToolObservation] = []
    for item in snapshot.get("recent_transitions") or []:
        if not isinstance(item, dict):
            continue
        observations.append(
            ToolObservation(
                tool=str(item.get("tool") or ""),
                tool_input=_compact(item.get("tool_input")),
                summary=_strip_learning_guidance(item.get("observation_summary")),
            )
        )
    return observations


def observations_from_messages(messages: Sequence[Any] | None) -> list[ToolObservation]:
    """Extract tool observations from LangChain message trajectories."""

    if not messages:
        return []
    observations: list[ToolObservation] = []
    pending_names: dict[str, str] = {}
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            if call_id and name:
                pending_names[call_id] = name
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        message_type = getattr(message, "type", "")
        if not tool_call_id and message_type != "tool":
            continue
        name = str(getattr(message, "name", "") or pending_names.get(tool_call_id, ""))
        observations.append(
            ToolObservation(
                tool=name,
                tool_input="",
                summary=_strip_learning_guidance(getattr(message, "content", "")),
            )
        )
    return observations


def evaluate_fault_family_evidence(
    *,
    task_description: str,
    diagnosis_report: str,
    observations: Sequence[ToolObservation] | None = None,
    available_tools: Iterable[str] | None = None,
) -> EvidenceGateResult:
    """Check whether a diagnosis has family-specific evidence from this run."""

    observations = tuple(observations or ())
    observed_tools = tuple(sorted({obs.tool for obs in observations if obs.tool}))
    observation_text = "\n".join(
        " ".join(
            part
            for part in (
                obs.tool,
                obs.tool_input,
                obs.summary,
            )
            if part
        )
        for obs in observations
    )
    families = infer_fault_families(task_description, diagnosis_report)
    missing: list[str] = []
    suggested_steps: list[str] = []

    claims_final = bool(FINAL_CLAIM_RE.search(str(diagnosis_report or "")))
    if claims_final and not observation_text.strip():
        missing.append(
            "The report makes a final anomaly/localization/RCA claim but no current tool observation is visible."
        )

    available = tuple(str(tool or "") for tool in (available_tools or ()))
    for family in families:
        if _family_satisfied(family, observations, observation_text):
            continue
        missing.append(
            f"{family.label}: missing {family.probes[0].description}."
        )
        tools = _available_probe_tools(family, available)
        if tools:
            suggested_steps.append(
                f"{family.label}: use {', '.join(tools[:5])} to collect {family.probes[0].description}."
            )
        else:
            suggested_steps.append(
                f"{family.label}: collect {family.probes[0].description} with the most specific available diagnostic tool."
            )

    sufficient = not missing
    prompt = (
        ""
        if sufficient
        else _remediation_prompt(
            families=families,
            missing=missing,
            suggested_steps=suggested_steps,
            observed_tools=observed_tools,
        )
    )
    return EvidenceGateResult(
        sufficient=sufficient,
        families=tuple(family.label for family in families),
        observed_tools=observed_tools,
        missing_evidence=tuple(missing),
        suggested_steps=tuple(suggested_steps),
        prompt=prompt,
    )


def evidence_gate_plan_steps(result: EvidenceGateResult, *, limit: int = 2) -> list[dict[str, str]]:
    """Convert a failed gate verdict into generic plan-step dictionaries."""

    steps: list[dict[str, str]] = []
    for index, step in enumerate(result.suggested_steps[:limit], start=1):
        steps.append(
            {
                "step_id": f"evidence_gate_{index}",
                "action": step,
                "expected_evidence": (
                    result.missing_evidence[index - 1]
                    if index - 1 < len(result.missing_evidence)
                    else "Family-specific discriminating evidence"
                ),
            }
        )
    return steps


def _family_satisfied(
    family: FaultFamily,
    observations: Sequence[ToolObservation],
    observation_text: str,
) -> bool:
    for probe in family.probes:
        if any(
            (obs.summary.strip() or obs.tool_input.strip())
            and _tool_matches(obs.tool, probe.tool_patterns)
            for obs in observations
        ):
            return True
        if any(re.search(pattern, observation_text, re.I) for pattern in probe.text_patterns):
            return True
    return False


def _tool_matches(tool_name: str, patterns: Sequence[str]) -> bool:
    normalized = str(tool_name or "").lower()
    return any(pattern.lower() in normalized for pattern in patterns)


def _available_probe_tools(
    family: FaultFamily,
    available_tools: Sequence[str],
) -> list[str]:
    if not available_tools:
        return []
    seen: set[str] = set()
    tools: list[str] = []
    for probe in family.probes:
        for pattern in probe.tool_patterns:
            for tool in available_tools:
                if _tool_matches(tool, (pattern,)) and tool not in seen:
                    seen.add(tool)
                    tools.append(tool)
    return tools


def _remediation_prompt(
    *,
    families: Sequence[FaultFamily],
    missing: Sequence[str],
    suggested_steps: Sequence[str],
    observed_tools: Sequence[str],
) -> str:
    families_text = ", ".join(family.label for family in families) or "unspecified"
    observed = ", ".join(observed_tools) if observed_tools else "none"
    return (
        "Evidence gate blocked finalization for this diagnosis.\n"
        f"Fault families implicated by the task/report: {families_text}.\n"
        f"Observed tools so far: {observed}.\n"
        "Missing evidence:\n"
        + "\n".join(f"- {item}" for item in missing)
        + "\nSuggested next checks:\n"
        + "\n".join(f"- {item}" for item in suggested_steps)
        + "\nInstructions:\n"
        "- Do not submit or restate the same final diagnosis yet.\n"
        "- Call diagnostic tools now to collect the missing discriminating evidence.\n"
        "- Broad reachability or task wording can support detection, but not localization or RCA by itself.\n"
        "- After the extra tool observations, produce a concise final report with anomaly status, faulty devices, root cause, and cited observations."
    )


def _strip_learning_guidance(value: Any) -> str:
    text = str(value or "")
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    return text.strip()


def _compact(value: Any, *, limit: int = 500) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
