"""Skill-Pro procedural memory service.

This adapts the official Skill-Pro semantics to NIKA's diagnosis-agent
boundary: a Skill-MDP style selector injects active procedural skills before
diagnosis, while closed benchmark episodes feed an ExperiencePool /
GoldenExperiencePool and non-parametric PPO-style evolution gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

from agent.learning_llm import (
    format_learning_error,
    learning_backend,
    learning_max_retries,
    learning_model,
    learning_timeout_seconds,
)
from agent.llm.model_factory import load_model
from agent.memory.safety import redact_oracle_markers
from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import (
    EvaluationEvidence,
    MemoryQuery,
    PPOGateDecision,
    ProceduralSkill,
    SemanticGradient,
    SemanticGradientDraft,
    SkillComponentGradient,
    SkillExperience,
    SkillRetrieval,
    SkillStep,
    SkillTransition,
    utc_now,
)
from agent.memory.store import SkillMemoryStore, public_episode_evidence

DEFAULT_POOL_SIZE = 32
EXPERIENCE_POOL_SIZE = 1000
GOLDEN_POOL_SIZE = 20
PPO_EPSILON = 0.2

GENERIC_SEED_SKILL_IDS = frozenset(
    {
        "seed_structured_cot",
        "seed_react_decision",
        "seed_hypothesis_elimination",
        "seed_self_consistency_check",
        "seed_explore_exploit",
        "seed_strategic_planning",
    }
)
EVIDENCE_LADDER_SEED_SKILL_IDS = frozenset(
    {
        "seed_name_resolution_ladder",
        "seed_host_addressing_ladder",
        "seed_routing_adjacency_ladder",
        "seed_bgp_config_disambiguation",
    }
)
SEED_SKILL_IDS = GENERIC_SEED_SKILL_IDS | EVIDENCE_LADDER_SEED_SKILL_IDS
BASELINE_EMA_ALPHA = 0.1


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _metric_success(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("detection_score") or 0) >= 1.0
        and _component_complete(metrics, "localization")
        and _component_complete(metrics, "rca")
    )


def _safe_skill_promotion(metrics: dict[str, Any]) -> bool:
    return _metric_success(metrics)


def _partial_outcome(metrics: dict[str, Any]) -> bool:
    detection = float(metrics.get("detection_score") or 0.0)
    return detection >= 1.0 and not (
        _component_complete(metrics, "localization")
        and _component_complete(metrics, "rca")
    )


def _component_complete(metrics: dict[str, Any], prefix: str) -> bool:
    accuracy = float(metrics.get(f"{prefix}_accuracy") or 0.0)
    f1 = float(metrics.get(f"{prefix}_f1") or 0.0)
    return max(accuracy, f1) >= 1.0


def _component_reward(metrics: dict[str, Any], prefix: str) -> float:
    accuracy = float(metrics.get(f"{prefix}_accuracy") or 0.0)
    precision = float(metrics.get(f"{prefix}_precision") or 0.0)
    f1 = float(metrics.get(f"{prefix}_f1") or 0.0)
    partial = (0.7 * f1) + (0.3 * precision)
    return max(accuracy, partial)


def _evidence_score(evidence: EvaluationEvidence) -> float:
    detection = float(evidence.metrics.get("detection_score") or 0.0)
    localization = _component_reward(evidence.metrics, "localization")
    rca = _component_reward(evidence.metrics, "rca")
    if detection <= 0 or (localization <= 0 and rca <= 0):
        accuracy = 0.0
    else:
        diagnosis_quality = detection * 0.15 + localization * 0.35 + rca * 0.5
        balance = 0.5 + 0.5 * min(localization, rca)
        accuracy = diagnosis_quality * balance
    step_penalty = min(evidence.steps or 0, 100) / 250.0
    tool_penalty = min(evidence.tool_calls or 0, 200) / 500.0
    return max(0.0, accuracy - step_penalty - tool_penalty)


def _skill_stat_reward(evidence: EvaluationEvidence, reward: float, baseline: float) -> tuple[float, float]:
    """Return reward/baseline used for online Skill-Pro score maintenance.

    Skill-Pro's online score should preserve reusable gains, not merely credit a
    skill for detecting that something is wrong. Episodes with incomplete
    localization/RCA become negative evidence for any reused skill.
    """

    if _safe_skill_promotion(evidence.metrics):
        return reward, baseline
    penalty = 0.08 if _partial_outcome(evidence.metrics) else 0.12
    return min(reward, baseline - penalty), baseline


def _evidence_signature_text(attrs: Any, tool_steps: list[SkillStep]) -> str:
    labels = attrs.protocols[:2] + attrs.services[:2] + attrs.symptoms[:3]
    tools = [step.tool_name for step in tool_steps if step.tool_name][:4]
    pieces: list[str] = []
    if labels:
        pieces.append("evidence labels: " + ", ".join(labels))
    if tools:
        pieces.append("observed tools: " + ", ".join(tools))
    return "; ".join(pieces) or "matching current observations"


def _trim_text(value: Any, *, limit: int = 360) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _compact_value(value: Any, *, limit: int = 240) -> Any:
    if isinstance(value, str):
        return _trim_text(value, limit=limit)
    if isinstance(value, list):
        return [_compact_value(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, limit=limit)
            for key, item in list(value.items())[:12]
        }
    return value


def _skill_steps_summary(tool_steps: list[SkillStep]) -> list[dict[str, Any]]:
    return [
        {
            "order": step.order,
            "action": _trim_text(step.action, limit=220),
            "skill_id": step.skill_id,
            "tool_name": step.tool_name,
            "arguments_hint": _compact_value(step.arguments_hint, limit=160),
            "status": step.status,
            "observation_summary": _trim_text(step.observation_summary, limit=240),
            "rationale": _trim_text(step.rationale, limit=160),
        }
        for step in tool_steps[:8]
    ]


def _skill_topic(
    evidence: EvaluationEvidence,
    attrs_protocols: list[str],
    attrs_services: list[str],
    attrs_symptoms: list[str],
) -> str:
    pieces = attrs_protocols[:2] + attrs_services[:2] + attrs_symptoms[:2]
    if pieces:
        return ", ".join(pieces)
    return evidence.scenario or evidence.topology_class or "network diagnosis"


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _jaccard(left: str, right: str) -> float:
    lhs = _tokens(left)
    rhs = _tokens(right)
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def _dump_for_alignment(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _skill_base_id(skill_id: str) -> str:
    return re.sub(r"_v\d+(?:_[a-f0-9]{6})?$", "", skill_id)


def _is_seed_skill(skill: ProceduralSkill) -> bool:
    return skill.skill_id in SEED_SKILL_IDS


def _uses_generic_seed_policy(skill: ProceduralSkill | None) -> bool:
    return bool(skill and _skill_base_id(skill.skill_id) in GENERIC_SEED_SKILL_IDS)


def _learned_skill_unstable(skill: ProceduralSkill) -> bool:
    if _is_seed_skill(skill) or skill.status == "retired":
        return False
    observed = skill.success_count + skill.failure_count
    if observed < 5 or skill.failure_count < 3:
        return False
    failure_rate = skill.failure_count / max(observed, 1)
    weak_score = max(skill.score, skill.avg_gain) < 0.4
    negative_gain = skill.avg_gain < 0.0
    repeated_failures = skill.failure_count >= 8 and failure_rate >= 0.6
    return repeated_failures or (failure_rate >= 0.6 and (weak_score or negative_gain))


def _has_name_resolution_signal(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\bdns\b",
            r"\bresolv(?:e|er|ing|ution)\b",
            r"\bname[-_ ]?lookup\b",
            r"\bnameserver\b",
            r"\bnslookup\b",
            r"\bdig\b",
            r"\bservfail\b",
            r"\bnxdomain\b",
            r"\bport\s+53\b",
        )
    )


def _has_host_addressing_signal(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\bdhcp\b",
            r"\blease\b",
            r"\bdefault\s+(?:via|route|gateway)\b",
            r"\bgateway\b",
            r"\bmissing\s+ip\b",
            r"\bno\s+(?:ip|inet)\b",
            r"\bip_route\s+is\s+empty\b",
            r"\bget_host_net_config\(",
            r"\bdhclient\b",
        )
    )


def _has_routing_adjacency_signal(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in (
            r"\bospf\b",
            r"\bfrr_get_ospf_conf\(",
            r"\bshow\s+ip\s+ospf\b",
            r"\brouting\s+adjacency\b",
            r"\bneighbor\s+(?:down|state|full|init|exstart|2-way)\b",
            r"\broute\s+missing\b",
            r"\bmissing\s+route\b",
            r"\bno\s+route\b",
            r"\bfrr_show_ip_route\(",
        )
    )


def _seed_activation_blocked(skill: ProceduralSkill, query_text: str) -> bool:
    text = str(query_text or "").lower()
    if skill.skill_id == "seed_name_resolution_ladder":
        return not _has_name_resolution_signal(text)
    if skill.skill_id == "seed_host_addressing_ladder":
        return not _has_host_addressing_signal(text)
    if skill.skill_id == "seed_routing_adjacency_ladder":
        bgp_only = "bgp" in text and not any(
            marker in text for marker in ("ospf", "frr_get_ospf_conf", "show ip ospf")
        )
        return bgp_only or not _has_routing_adjacency_signal(text)
    if skill.skill_id != "seed_bgp_config_disambiguation":
        return False
    endpoint_checked = any(
        marker in text
        for marker in (
            "get_host_net_config(",
            "ethtool(",
            "ip_addr_statistics(",
            '"ip_addr"',
            '"ifconfig"',
            "link detected:",
            "endpoint host link checked",
            "endpoint checks",
            "host link checked",
        )
    )
    bgp_observation_context = any(
        marker in text
        for marker in (
            "neighbor",
            "state/pfxrcd",
            "frr_show_bgp_summary(",
            "frr_show_ip_route(",
            "frr_get_bgp_conf(",
            "frr_show_running_config(",
        )
    )
    bgp_state_abnormal = bool(re.search(r"\b(idle|active|connect)\b", text)) and (
        bgp_observation_context
    )
    route_or_config_abnormal = any(
        marker in text
        for marker in (
            "remote-as",
            "as mismatch",
            "missing route",
            "no route",
            "network is unreachable",
            "not advertised",
            "prefixes received 0",
            "pfxrcd 0",
            "rib-failure",
        )
    ) and bgp_observation_context
    bgp_abnormal = bgp_state_abnormal or route_or_config_abnormal
    return not (endpoint_checked and bgp_abnormal)


def _activation_similarity(skill: ProceduralSkill, query: MemoryQuery) -> float:
    state_text = " ".join(
        [
            query.text,
            " ".join(query.protocols),
            " ".join(query.services),
            " ".join(query.symptoms),
        ]
    )
    activation = skill.activation_condition
    lexical = _jaccard(state_text, activation)
    query_tokens = _tokens(state_text)
    activation_tokens = _tokens(activation)
    scope_tokens = set(skill.protocols) | set(skill.services) | set(skill.symptoms)
    scope_hits = len(query_tokens & scope_tokens) / max(len(scope_tokens), 1) if scope_tokens else 0.0
    discriminators = {
        token
        for token in activation_tokens
        if len(token) >= 5 and token not in {"current", "evidence", "diagnosis", "symptoms"}
    }
    discriminator_hits = (
        len(query_tokens & discriminators) / max(len(discriminators), 1)
        if discriminators
        else 0.0
    )
    return _clamp((0.55 * lexical) + (0.3 * scope_hits) + (0.15 * discriminator_hits), 0.0, 1.0)


def _signature_activation(signature: str) -> str:
    return (
        "Use when the current observation history matches this "
        f"evidence signature: {signature}. Do not activate from "
        "scenario name or tool catalog alone."
    )


def _experience_signature(exp: SkillExperience) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    tools = [step.tool_name for step in exp.transitions if step.tool_name]
    text = " ".join(
        [
            exp.trajectory,
            " ".join(
                " ".join(
                    str(item or "")
                    for item in (
                        step.action,
                        step.tool_name,
                        _dump_for_alignment(step.arguments_hint),
                        step.observation_summary,
                    )
                )
                for step in exp.transitions
            ),
        ]
    )
    attrs = infer_memory_attributes(text, scenario=exp.scenario, tools=tools)
    return (
        frozenset(attrs.protocols),
        frozenset(attrs.services),
        frozenset(attrs.symptoms),
    )


def _compatible_experience_signature(
    left: tuple[frozenset[str], frozenset[str], frozenset[str]],
    right: tuple[frozenset[str], frozenset[str], frozenset[str]],
) -> bool:
    left_protocols, left_services, left_symptoms = left
    right_protocols, right_services, right_symptoms = right
    if left_protocols and right_protocols:
        return bool(left_protocols & right_protocols)
    return bool((left_services & right_services) or (left_symptoms & right_symptoms))


def _episode_attribute_text(
    evidence: EvaluationEvidence,
    tool_steps: list[SkillStep],
) -> str:
    step_text = " ".join(
        " ".join(
            str(item or "")
            for item in (
                step.action,
                step.tool_name,
                step.arguments_hint,
                step.observation_summary,
            )
        )
        for step in tool_steps
    )
    return " ".join([evidence.task_description, step_text])


def _redact_hidden_labels(text: str, evidence: EvaluationEvidence) -> str:
    redacted = text
    for label in [*evidence.root_cause, *evidence.faulty_devices]:
        if not label:
            continue
        redacted = re.sub(re.escape(label), "[redacted]", redacted, flags=re.IGNORECASE)
    return redacted


class ProceduralMemoryModule:
    def __init__(
        self,
        *,
        bank_id: str = "default",
        llm_backend: str | None = None,
        model: str | None = None,
        store_path: Path | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        evolution_threshold: int = 3,
        best_of_n: int = 3,
        ppo_epsilon: float = PPO_EPSILON,
    ) -> None:
        self.bank_id = bank_id
        self.llm_backend = llm_backend
        self.model = model
        self.pool_size = pool_size
        self.evolution_threshold = evolution_threshold
        self.best_of_n = max(1, best_of_n)
        self.ppo_epsilon = ppo_epsilon
        self.store = SkillMemoryStore(
            bank_id=bank_id,
            root=store_path.parent if store_path else None,
        )
        self._ensure_seed_skills()

    def clear(self) -> None:
        self.store.clear()
        self._ensure_seed_skills()

    def _ensure_seed_skills(self) -> None:
        state = self.store.load()
        changed = False
        for skill in self._seed_skills():
            if skill.skill_id not in state.skills:
                state.skills[skill.skill_id] = skill
                changed = True
        if changed:
            self.store.save(state)

    def _seed_skills(self) -> list[ProceduralSkill]:
        seed_specs = [
            (
                "seed_structured_cot",
                "StructuredCoT",
                "When a decision must be made from multiple constraints or conflicting evidence.",
                [
                    "Restate the immediate diagnostic goal.",
                    "List hard constraints and observed evidence.",
                    "Compare candidate fault classes under those constraints.",
                    "Choose the next action that best reduces uncertainty.",
                ],
                "Stop after one concrete diagnostic action or hypothesis is selected.",
            ),
            (
                "seed_react_decision",
                "ReActDecision",
                "When tool feedback should directly influence the next diagnostic action.",
                [
                    "Interpret the latest tool feedback.",
                    "Update the belief about likely faulty components.",
                    "Choose the next tool/action that tests the updated belief.",
                ],
                "Stop after selecting the next evidence-gathering action.",
            ),
            (
                "seed_hypothesis_elimination",
                "HypothesisElimination",
                "When several root-cause hypotheses remain plausible.",
                [
                    "Enumerate plausible hypotheses.",
                    "Remove hypotheses contradicted by observations.",
                    "Identify the most discriminating missing evidence.",
                    "Collect that evidence before committing to RCA.",
                ],
                "Stop when only one supported hypothesis remains or evidence budget is exhausted.",
            ),
            (
                "seed_self_consistency_check",
                "SelfConsistencyCheck",
                "When a diagnosis or action must satisfy known topology and evidence constraints.",
                [
                    "Draft the candidate diagnosis.",
                    "Check it against topology, reachability, and service evidence.",
                    "Revise if any observation contradicts it.",
                ],
                "Stop when the diagnosis is internally consistent with collected evidence.",
            ),
            (
                "seed_explore_exploit",
                "ExploreExploitArbitration",
                "When deciding between broad exploration and exploiting a strong current hypothesis.",
                [
                    "Estimate whether current evidence is sufficient.",
                    "Explore if uncertainty remains high.",
                    "Exploit by verifying the leading hypothesis if confidence is high.",
                ],
                "Stop after choosing exploration or exploitation for the next action.",
            ),
            (
                "seed_strategic_planning",
                "StrategicPlanning",
                "At the beginning of a diagnosis with little or no evidence.",
                [
                    "Identify the task family and likely protocol/service layer.",
                    "Plan a short evidence ladder from broad health checks to specific RCA checks.",
                    "Prioritize low-cost tools before disruptive commands.",
                ],
                "Stop after creating the initial diagnostic plan.",
            ),
        ]
        skills = [
            ProceduralSkill(
                skill_id=skill_id,
                title=title,
                activation_condition=activation,
                execution_steps=[
                    SkillStep(order=index + 1, action=action)
                    for index, action in enumerate(policy)
                ],
                termination_condition=termination,
                status="validated",
                score=0.25,
            )
            for skill_id, title, activation, policy, termination in seed_specs
        ]
        skills.extend(
            [
                ProceduralSkill(
                    skill_id="seed_name_resolution_ladder",
                    title="NameResolutionEvidenceLadder",
                    activation_condition=(
                        "When the current claim or observations involve DNS, "
                        "resolver, name lookup, nameserver, or port 53 symptoms."
                    ),
                    execution_steps=[
                        SkillStep(
                            order=1,
                            action=(
                                "Confirm the user-visible lookup symptom with a "
                                "direct resolution or HTTP timing check from an "
                                "affected endpoint."
                            ),
                            tool_name="curl_web_test",
                        ),
                        SkillStep(
                            order=2,
                            action=(
                                "Inspect resolver configuration or the queried "
                                "nameserver before blaming downstream HTTP, ACL, "
                                "or routing components."
                            ),
                            tool_name="cat_file",
                        ),
                        SkillStep(
                            order=3,
                            action=(
                                "Verify the name service process and listener "
                                "state on the suspected server."
                            ),
                            tool_name="systemctl_ops",
                        ),
                    ],
                    termination_condition=(
                        "Stop when direct lookup/config/service evidence either "
                        "supports a name-resolution RCA or rules it out."
                    ),
                    protocols=["dns"],
                    services=["name_resolution"],
                    symptoms=["lookup_failure", "service_unreachable", "latency"],
                    tools=[
                        "curl_web_test",
                        "cat_file",
                        "systemctl_ops",
                        "netstat",
                        "exec_shell",
                    ],
                    status="validated",
                    score=0.16,
                ),
                ProceduralSkill(
                    skill_id="seed_host_addressing_ladder",
                    title="HostAddressingEvidenceLadder",
                    activation_condition=(
                        "When the current claim or observations involve DHCP, "
                        "lease, host IP, default route, gateway, or address "
                        "configuration symptoms."
                    ),
                    execution_steps=[
                        SkillStep(
                            order=1,
                            action=(
                                "Inspect the affected endpoint's IP address, "
                                "default route, and resolver state."
                            ),
                            tool_name="get_host_net_config",
                        ),
                        SkillStep(
                            order=2,
                            action=(
                                "Compare against a healthy endpoint or expected "
                                "subnet before blaming routers or dynamic routing."
                            ),
                            tool_name="get_host_net_config",
                        ),
                        SkillStep(
                            order=3,
                            action=(
                                "If host addressing is abnormal, inspect DHCP "
                                "service/config/lease evidence on the provider."
                            ),
                            tool_name="systemctl_ops",
                        ),
                    ],
                    termination_condition=(
                        "Stop when current host addressing and DHCP evidence "
                        "supports one addressing RCA or rules the layer out."
                    ),
                    protocols=["dhcp"],
                    services=["addressing"],
                    symptoms=["missing_ip", "bad_gateway", "lease_failure"],
                    tools=[
                        "get_host_net_config",
                        "systemctl_ops",
                        "cat_file",
                        "exec_shell",
                    ],
                    status="validated",
                    score=0.16,
                ),
                ProceduralSkill(
                    skill_id="seed_routing_adjacency_ladder",
                    title="RoutingAdjacencyEvidenceLadder",
                    activation_condition=(
                        "When the current claim or observations involve IGP "
                        "routing adjacency, route propagation, OSPF, or FRR "
                        "route symptoms after endpoint checks."
                    ),
                    execution_steps=[
                        SkillStep(
                            order=1,
                            action=(
                                "Confirm the missing or wrong route on the "
                                "affected router."
                            ),
                            tool_name="frr_show_ip_route",
                        ),
                        SkillStep(
                            order=2,
                            action=(
                                "Inspect routing neighbor state on the affected "
                                "router and a directly connected peer."
                            ),
                            tool_name="frr_exec",
                        ),
                        SkillStep(
                            order=3,
                            action=(
                                "Inspect routing configuration only after route "
                                "or adjacency evidence points to the control plane."
                            ),
                            tool_name="frr_get_ospf_conf",
                        ),
                    ],
                    termination_condition=(
                        "Stop when route, adjacency, and config evidence support "
                        "one routing-control-plane RCA or rule it out."
                    ),
                    protocols=["ospf"],
                    services=["routing"],
                    symptoms=["missing_route", "neighbor_down", "control_plane"],
                    tools=[
                        "frr_show_ip_route",
                        "frr_exec",
                        "frr_get_ospf_conf",
                        "systemctl_ops",
                    ],
                    status="validated",
                    score=0.16,
                ),
                ProceduralSkill(
                    skill_id="seed_bgp_config_disambiguation",
                    title="BGPConfigDisambiguation",
                    activation_condition=(
                        "When endpoint host/link state has been checked and current "
                        "BGP evidence still shows neighbor Idle, Active, Connect, "
                        "remote-as mismatch, or missing route/control-plane symptoms."
                    ),
                    execution_steps=[
                        SkillStep(
                            order=1,
                            action=(
                                "Confirm the abnormal BGP neighbor or route symptom "
                                "with frr_show_bgp_summary or frr_show_ip_route."
                            ),
                            tool_name="frr_show_bgp_summary",
                        ),
                        SkillStep(
                            order=2,
                            action=(
                                "Inspect the running BGP configuration on the affected "
                                "router."
                            ),
                            tool_name="frr_get_bgp_conf",
                        ),
                        SkillStep(
                            order=3,
                            action=(
                                "Inspect the directly connected peer's BGP "
                                "configuration and compare neighbor, remote-as, and "
                                "network advertisement statements."
                            ),
                            tool_name="frr_show_running_config",
                        ),
                        SkillStep(
                            order=4,
                            action=(
                                "Commit to service-down, missing-advertisement, or "
                                "AS/configuration RCA only after the current config "
                                "evidence supports it."
                            ),
                        ),
                    ],
                    termination_condition=(
                        "Stop when current BGP configuration evidence supports one "
                        "remaining control-plane/configuration hypothesis, or when "
                        "new endpoint evidence contradicts the activation condition."
                    ),
                    protocols=["bgp"],
                    services=["routing"],
                    symptoms=["missing_route", "neighbor_down", "control_plane"],
                    tools=[
                        "frr_show_bgp_summary",
                        "frr_show_ip_route",
                        "frr_get_bgp_conf",
                        "frr_show_running_config",
                    ],
                    status="validated",
                    score=0.18,
                ),
            ]
        )
        return skills

    def retrieve(self, *, query: MemoryQuery, session_id: str = "") -> list[SkillRetrieval]:
        state = self.store.load()
        scored: list[SkillRetrieval] = []
        query_text = " ".join(
            [
                query.text,
                query.scenario,
                " ".join(query.protocols),
                " ".join(query.services),
                " ".join(query.symptoms),
                " ".join(query.tools),
            ]
        ).lower()
        total_maturity = max([skill.maturity for skill in state.skills.values()] or [1])
        for skill in state.skills.values():
            if skill.status == "retired":
                continue
            if _learned_skill_unstable(skill):
                continue
            if _seed_activation_blocked(skill, query.text):
                continue
            reasons: list[str] = []
            score = self._skill_effective_score(skill)
            scope_delta, scope_reasons, scope_blocked = self._transfer_scope_adjustment(
                skill=skill,
                query=query,
            )
            if scope_blocked:
                continue
            score += scope_delta
            reasons.extend(scope_reasons)
            activation_fit = _activation_similarity(skill, query)
            if not _is_seed_skill(skill) and activation_fit < 0.03 and (
                skill.protocols or skill.services or skill.symptoms
            ):
                continue
            score += 0.35 * activation_fit
            if activation_fit > 0:
                reasons.append(f"activation_fit:{activation_fit:.2f}")
            if query.scenario and skill.scenarios:
                if query.scenario in skill.scenarios:
                    score += 0.2
                    reasons.append(f"scenario:{query.scenario}")
                else:
                    score -= 0.35
            for label, values in (
                ("protocol", skill.protocols),
                ("service", skill.services),
                ("symptom", skill.symptoms),
                ("tool", skill.tools),
            ):
                query_values = getattr(query, f"{label}s", []) if label != "tool" else query.tools
                overlap = set(values) & set(query_values)
                if overlap:
                    score += 0.15 * len(overlap)
                    reasons.append(f"{label}:{','.join(sorted(overlap))}")
                elif query_values and values:
                    score -= 0.12 if label in {"protocol", "tool"} else 0.06
            for token in skill.activation_condition.lower().split():
                if len(token) > 3 and token in query_text:
                    score += 0.01
            score += 0.1 * self._lcb_bonus(skill, total_maturity)
            if score > 0:
                scored.append(SkillRetrieval(memory=skill, score=score, reasons=reasons))
        scored.sort(key=lambda item: item.score, reverse=True)
        selected: list[SkillRetrieval] = []
        used_tokens = 0
        for item in scored:
            if len(selected) >= query.top_k:
                break
            cost = _estimate_tokens(item.skill.format_for_llm())
            if selected and used_tokens + cost > query.token_budget:
                continue
            selected.append(item)
            used_tokens += cost
        return selected

    def select_skill(
        self,
        *,
        query: MemoryQuery,
        session_id: str = "",
        top_k: int | None = None,
        min_lcb: float = -0.05,
        record_reuse: bool = True,
        exclude_skill_ids: Collection[str] | None = None,
        allow_excluded_fallback: bool = True,
    ) -> SkillRetrieval | None:
        top_k = top_k or query.top_k
        excluded = {skill_id for skill_id in (exclude_skill_ids or []) if skill_id}
        candidates = self.retrieve(
            query=query.model_copy(
                update={"top_k": max(top_k + len(excluded), 1)}
            ),
            session_id=session_id,
        )
        if not candidates:
            return None
        selectable = [
            item for item in candidates if item.skill.skill_id not in excluded
        ]
        if not selectable and not allow_excluded_fallback:
            return None
        pool = selectable or candidates
        selected = next(
            (
                item
                for item in pool
                if item.skill.maturity < 3 or self._lcb(item.skill) >= min_lcb
            ),
            None,
        )
        if selected is None:
            return None
        if record_reuse:
            selected = self._record_skill_reuse(selected)
        return selected

    def _transfer_scope_adjustment(
        self,
        *,
        skill: ProceduralSkill,
        query: MemoryQuery,
    ) -> tuple[float, list[str], bool]:
        if _is_seed_skill(skill):
            return 0.0, [], False
        reasons: list[str] = []
        delta = 0.0
        if query.scenario and skill.scenarios and query.scenario not in skill.scenarios:
            delta -= 0.12
            reasons.append("scenario_mismatch_metadata")

        query_scope = {
            "protocol": set(query.protocols),
            "service": set(query.services),
            "symptom": set(query.symptoms),
        }
        skill_scope = {
            "protocol": set(skill.protocols),
            "service": set(skill.services),
            "symptom": set(skill.symptoms),
        }
        discriminating_query_labels = set().union(*query_scope.values())
        discriminating_skill_labels = set().union(*skill_scope.values())
        if discriminating_skill_labels and not discriminating_query_labels:
            return -0.45, ["needs_current_evidence_signature"], True

        overlap_count = 0
        mismatch_count = 0
        for label, skill_values in skill_scope.items():
            query_values = query_scope[label]
            if not skill_values or not query_values:
                continue
            overlap = skill_values & query_values
            if overlap:
                overlap_count += len(overlap)
                delta += 0.12 * len(overlap)
                reasons.append(f"scope_{label}:{','.join(sorted(overlap))}")
            else:
                mismatch_count += 1
                if label == "symptom":
                    return 0.0, [], True
                delta -= 0.18
        if discriminating_skill_labels and discriminating_query_labels and overlap_count == 0:
            return 0.0, [], True

        query_tools = set(query.tools)
        skill_tools = set(skill.tools)
        if skill_tools and query_tools:
            tool_overlap = skill_tools & query_tools
            if tool_overlap:
                delta += min(0.12, 0.04 * len(tool_overlap))
                reasons.append(f"scope_tool:{','.join(sorted(tool_overlap)[:3])}")
            elif mismatch_count:
                delta -= 0.08
        return delta, reasons, False

    def select_skill_llm_topk_lcb(
        self,
        *,
        query: MemoryQuery,
        llm_agent: Any,
        session_id: str = "",
        top_k: int | None = None,
        nominee_k: int = 3,
        min_lcb: float = -0.05,
        record_reuse: bool = True,
        exclude_skill_ids: Collection[str] | None = None,
        allow_excluded_fallback: bool = True,
    ) -> SkillRetrieval | None:
        """Skill-Pro style selector: LLM nominates top-k, LCB picks one."""
        top_k = top_k or query.top_k
        excluded = {skill_id for skill_id in (exclude_skill_ids or []) if skill_id}
        candidates = self.retrieve(
            query=query.model_copy(
                update={"top_k": max(top_k + len(excluded), nominee_k, 1)}
            ),
            session_id=session_id,
        )
        if not candidates:
            return None
        selectable = [
            item for item in candidates if item.skill.skill_id not in excluded
        ]
        if not selectable and not allow_excluded_fallback:
            return None
        pool = selectable or candidates
        choices = self._llm_skill_nominees(
            query=query,
            candidates=pool,
            llm_agent=llm_agent,
            nominee_k=nominee_k,
        )
        if choices is None:
            return self.select_skill(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_lcb=min_lcb,
                record_reuse=record_reuse,
                exclude_skill_ids=exclude_skill_ids,
                allow_excluded_fallback=allow_excluded_fallback,
            )
        if not choices:
            return self.select_skill(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_lcb=min_lcb,
                record_reuse=record_reuse,
                exclude_skill_ids=exclude_skill_ids,
                allow_excluded_fallback=allow_excluded_fallback,
            )
        lookup: dict[str, SkillRetrieval] = {}
        for item in pool:
            skill = item.skill
            lookup[skill.skill_id.lower()] = item
            lookup[skill.title.lower()] = item
        nominated: list[SkillRetrieval] = []
        seen: set[str] = set()
        for choice in choices:
            item = lookup.get(choice.lower())
            if item is None or item.skill.skill_id in seen:
                continue
            nominated.append(item)
            seen.add(item.skill.skill_id)
        if not nominated:
            return self.select_skill(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_lcb=min_lcb,
                record_reuse=record_reuse,
                exclude_skill_ids=exclude_skill_ids,
                allow_excluded_fallback=allow_excluded_fallback,
            )
        ranked = sorted(
            nominated,
            key=lambda item: self._skill_lcb_from_current_state(item.skill),
            reverse=True,
        )
        selected = next(
            (
                item
                for item in ranked
                if item.skill.maturity < 3 or self._lcb(item.skill) >= min_lcb
            ),
            None,
        )
        if selected is None:
            return None
        if record_reuse:
            selected = self._record_skill_reuse(selected)
        return selected

    def _llm_skill_nominees(
        self,
        *,
        query: MemoryQuery,
        candidates: list[SkillRetrieval],
        llm_agent: Any,
        nominee_k: int,
    ) -> list[str] | None:
        skills_desc = "\n".join(
            (
                f"- {item.skill.skill_id} ({item.skill.title}) "
                f"score={item.score:.3f}: {item.skill.activation_condition}"
            )
            for item in candidates[: max(nominee_k * 2, nominee_k)]
        )
        prompt = (
            "You are the Skill-Pro skill selector for NIKA network diagnosis.\n\n"
            f"[CURRENT STATE]\n{query.text[:2500]}\n\n"
            f"[AVAILABLE SKILL-MDP OPTIONS]\n{skills_desc}\n- NONE\n\n"
            f"Select up to {nominee_k} skills that are most relevant and helpful. "
            "Prefer skills whose initiation condition fits the current state and "
            "whose policy can guide the next diagnostic tool call. Output only XML "
            "choice lines, for example:\n"
            "<choice>skill_id</choice>\n"
            "or <choice>NONE</choice>."
        )
        try:
            response = llm_agent.invoke(prompt)
            text = str(getattr(response, "content", response) or "")
        except Exception:
            return None
        names = [
            item.strip()
            for item in re.findall(r"<choice>\s*(.*?)\s*</choice>", text, re.I | re.S)
            if item.strip()
        ]
        if not names:
            return None
        if any(name.upper() == "NONE" for name in names):
            return []
        return names[: max(1, nominee_k)]

    def _record_skill_reuse(self, selected: SkillRetrieval) -> SkillRetrieval:
        state = self.store.load()
        stored = state.skills.get(selected.skill.skill_id)
        if stored is not None:
            stored.reuse_count += 1
            stored.updated_at = utc_now()
            state.skills[stored.skill_id] = stored
            self.store.save(state)
            selected.memory = stored
        return selected

    def format_context(
        self,
        retrieved: list[SkillRetrieval],
        *,
        active_skill_id: str | None = None,
    ) -> str:
        if not retrieved:
            return ""
        blocks = [
            "Retrieved Skill-Pro Skill-MDP procedures. Treat them as reusable diagnostic policies, not as ground truth."
        ]
        for index, item in enumerate(retrieved):
            skill = item.skill
            if active_skill_id is None:
                label = "ACTIVE" if index == 0 else "CANDIDATE"
            else:
                label = "ACTIVE" if skill.skill_id == active_skill_id else "CANDIDATE"
            blocks.append(
                redact_oracle_markers(
                    "\n".join(
                        [
                            f"- {label} Skill {skill.skill_id} ({skill.title}) score={item.score:.3f}",
                            f"  Activation / Initiation: {skill.activation_condition}",
                            "  Policy:",
                            *[
                                f"    {step.order}. {step.action}"
                                for step in skill.execution_steps[:6]
                            ],
                            f"  Termination: {skill.termination_condition}",
                        ]
                    )
                )
            )
        return "\n".join(blocks)

    def snapshot(self, *, session_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(self.store.snapshot_jsonl()) + "\n", encoding="utf-8")
        return output_path

    def propose_skill(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        parent: ProceduralSkill | None = None,
        candidate_index: int = 0,
        critique: SemanticGradient | None = None,
    ) -> ProceduralSkill:
        if not tool_steps:
            raise ValueError("Skill-Pro requires at least one observed execution step.")
        attrs = infer_memory_attributes(
            _episode_attribute_text(evidence, tool_steps),
            scenario=evidence.scenario,
            topology_class=evidence.topology_class,
            tools=[step.tool_name for step in tool_steps if step.tool_name],
        )
        topic = _skill_topic(evidence, attrs.protocols, attrs.services, attrs.symptoms)
        signature = _evidence_signature_text(attrs, tool_steps)
        critique = (
            critique.model_copy(deep=True)
            if critique is not None
            else self.semantic_gradient(evidence=evidence, tool_steps=tool_steps)
        )
        if parent is None:
            skill_id = _stable_id(
                attrs.protocols,
                attrs.services,
                attrs.symptoms,
                attrs.tools,
                [step.tool_name for step in tool_steps if step.tool_name],
                prefix="skill",
            )
            title = f"Procedure for {topic}"
            activation = _signature_activation(signature)
            steps = tool_steps[:10]
            termination = (
                "Stop when anomaly status, faulty devices, and root-cause class are supported "
                "by at least two independent observations, or when max diagnostic budget is reached."
            )
            version = 0
            parent_id = ""
        else:
            base = _skill_base_id(parent.skill_id)
            version = parent.version + 1
            revision = hashlib.sha256(
                json.dumps(
                    [
                        parent.skill_id,
                        evidence.session_id,
                        evidence.scenario,
                        attrs.protocols,
                        attrs.services,
                        attrs.symptoms,
                        attrs.tools,
                        candidate_index,
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()[:6]
            skill_id = f"{base}_v{version}_{revision}"
            title = parent.title
            activation = critique.component_update.initiation or (
                _signature_activation(signature)
                if _uses_generic_seed_policy(parent)
                else parent.activation_condition
            )
            update_steps = [
                SkillStep(order=i + 1, action=step, rationale="Skill-Pro semantic update.")
                for i, step in enumerate(critique.component_update.policy)
                if step.strip()
            ]
            steps = update_steps or parent.execution_steps
            termination = critique.component_update.termination or parent.termination_condition
            parent_id = parent.skill_id
        if critique.proposed_update:
            termination += f" Semantic update: {critique.proposed_update[:240]}"
        if candidate_index == 1:
            termination += " Require independent confirmation before final RCA."
        elif candidate_index >= 2:
            steps = steps + [
                SkillStep(
                    order=len(steps) + 1,
                    action="Cross-check the leading hypothesis with an independent tool before submitting.",
                    rationale="Best-of-N Skill-Pro candidate variant.",
                )
            ]
        outcome_success = _safe_skill_promotion(evidence.metrics)
        return ProceduralSkill(
            skill_id=skill_id,
            title=title,
            activation_condition=activation,
            execution_steps=steps[:10],
            termination_condition=termination,
            source_sessions=[evidence.session_id],
            scenarios=[evidence.scenario] if evidence.scenario else [],
            protocols=attrs.protocols,
            services=attrs.services,
            symptoms=attrs.symptoms,
            tools=attrs.tools,
            status="validated" if outcome_success else "candidate",
            success_count=1 if outcome_success else 0,
            failure_count=0 if outcome_success else 1,
            score=_evidence_score(evidence),
            parent_id=parent_id,
            version=version,
            semantic_gradients=[critique],
        )

    def semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        llm_gradient, llm_error = self._llm_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        if llm_gradient is not None:
            return llm_gradient
        gradient = self._deterministic_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        gradient.llm_error = llm_error
        return gradient

    def _deterministic_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        if _safe_skill_promotion(evidence.metrics):
            critique = "Successful trajectory: preserve evidence order and termination rule."
            update = "Promote or reinforce this procedure if it improves step/tool efficiency."
            component = SkillComponentGradient(
                policy=[step.action for step in tool_steps[:6]],
                termination="Terminate after direct evidence supports detection, localization, and RCA.",
                is_related=True,
            )
        else:
            detection = float(evidence.metrics.get("detection_score") or 0.0)
            high_tool_budget = (evidence.tool_calls or len(tool_steps)) >= 10
            if detection >= 1.0 and (
                not _component_complete(evidence.metrics, "localization")
                or not _component_complete(evidence.metrics, "rca")
            ):
                critique = (
                    "Failed Skill-Pro outcome: anomaly detection succeeded but "
                    "localization/RCA was not supported. Treat this as premature "
                    "termination or an overly broad initiation condition, not as "
                    "a reusable success."
                )
                update = (
                    "Narrow initiation to states with matching current evidence "
                    "signature and require discriminating localization/RCA evidence "
                    "before final diagnosis."
                )
                termination = (
                    "Terminate only after current observations support detection, "
                    "localization, and RCA; detection-only evidence must continue "
                    "to a discriminating check."
                )
            elif high_tool_budget:
                critique = (
                    "Failed Skill-Pro outcome: the procedure consumed too many "
                    "tool calls without converging. The termination condition is "
                    "too weak or the policy lacks a short evidence ladder."
                )
                update = (
                    "Add a bounded ladder and terminate or switch skills when the "
                    "latest observations no longer satisfy initiation."
                )
                termination = (
                    "Terminate or switch after the ladder's discriminating checks "
                    "are exhausted, or when observations contradict initiation."
                )
            else:
                critique = (
                    "Failed trajectory: revise initiation or policy to require stronger "
                    "evidence before localization/RCA."
                )
                update = "Store only as candidate unless PPO gate beats the existing/default policy."
                termination = "Do not terminate until diagnosis has at least two independent observations."
            component = SkillComponentGradient(
                initiation=(
                    "Use only when the current observation history matches the "
                    "skill's evidence signature; do not activate from scenario "
                    "or tool catalog alone."
                ),
                policy=[
                    "Identify the active evidence family from current observations.",
                    "Collect the most specific discriminating check for localization.",
                    "Verify the suspected root cause with an independent command.",
                ],
                termination=termination,
                is_related=bool(tool_steps),
            )
        if not tool_steps:
            critique += " Trace contained no usable diagnosis tool calls."
        return SemanticGradient(
            source_session_id=evidence.session_id,
            critique=critique,
            proposed_update=update,
            component_update=component,
            gradient_source="deterministic",
        )

    def _llm_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> tuple[SemanticGradient | None, str]:
        selected_backend = learning_backend(self.llm_backend)
        selected_model = learning_model(self.model)
        if not selected_backend or not selected_model:
            return None, ""
        metric_keys = (
            "detection_score",
            "localization_accuracy",
            "localization_precision",
            "localization_recall",
            "localization_f1",
            "rca_accuracy",
            "rca_precision",
            "rca_recall",
            "rca_f1",
            "steps",
            "tool_calls",
            "tool_errors",
        )
        public_evidence = {
            "source_session_id": evidence.session_id,
            "task_description": _trim_text(evidence.task_description, limit=900),
            "scenario": evidence.scenario,
            "topology_class": evidence.topology_class,
            "success": evidence.success,
            "steps": evidence.steps,
            "tool_calls": evidence.tool_calls,
            "metrics": {
                key: evidence.metrics.get(key)
                for key in metric_keys
                if key in evidence.metrics
            },
        }
        prompt = (
            "You are the Skill-Pro semantic-gradient critic for NIKA network diagnosis. "
            "Generate a short component-level update for a Skill-MDP option. Do not "
            "name hidden root causes or faulty devices.\n\n"
            f"Evaluation evidence:\n{json.dumps(public_evidence, indent=2, ensure_ascii=False)}\n\n"
            f"Observed execution steps:\n{json.dumps(_skill_steps_summary(tool_steps), indent=2, ensure_ascii=False)}\n\n"
            "Return a compact SemanticGradientDraft with critique, proposed_update, "
            "initiation, policy, termination, and is_related. Keep policy to at most "
            "four short steps. Use the same source_session_id."
        )
        try:
            llm = load_model(
                selected_backend,
                selected_model,
                timeout=learning_timeout_seconds(),
                max_retries=learning_max_retries(),
            )
            critic = llm.with_structured_output(SemanticGradientDraft)
            raw_gradient = critic.invoke(prompt)
            if isinstance(raw_gradient, SemanticGradient):
                gradient = raw_gradient
            else:
                draft = (
                    raw_gradient
                    if isinstance(raw_gradient, SemanticGradientDraft)
                    else SemanticGradientDraft.model_validate(raw_gradient)
                )
                gradient = SemanticGradient(
                    source_session_id=draft.source_session_id or evidence.session_id,
                    critique=draft.critique,
                    proposed_update=draft.proposed_update,
                    component_update=SkillComponentGradient(
                        initiation=draft.initiation,
                        policy=draft.policy[:4],
                        termination=draft.termination,
                        is_related=draft.is_related,
                    ),
                )
            if gradient.source_session_id != evidence.session_id:
                gradient.source_session_id = evidence.session_id
            gradient.critique = _redact_hidden_labels(gradient.critique, evidence)
            gradient.proposed_update = _redact_hidden_labels(gradient.proposed_update, evidence)
            gradient.component_update.initiation = _redact_hidden_labels(
                gradient.component_update.initiation,
                evidence,
            )
            gradient.component_update.policy = [
                _redact_hidden_labels(step, evidence)
                for step in gradient.component_update.policy
            ]
            gradient.component_update.termination = _redact_hidden_labels(
                gradient.component_update.termination,
                evidence,
            )
            gradient.gradient_source = "llm"
            gradient.llm_error = ""
            return gradient, ""
        except Exception as exc:
            return None, format_learning_error(exc)

    def ppo_gate(
        self,
        *,
        candidate: ProceduralSkill,
        evidence: EvaluationEvidence,
        baseline: ProceduralSkill | None = None,
        samples: list[SkillExperience] | None = None,
        candidate_type: str = "NEW",
        best_of_n: int = 1,
    ) -> PPOGateDecision:
        current_reward = _evidence_score(evidence)
        candidate_score = self._skill_effective_score(candidate)
        baseline_score = self._skill_effective_score(baseline) if baseline else 0.0
        promotion_safe = _safe_skill_promotion(evidence.metrics)
        sample_reward, sample_baseline = _skill_stat_reward(
            evidence,
            current_reward,
            baseline_score,
        )
        sample_batch = samples or [
            SkillExperience(
                experience_id=_stable_id(evidence.session_id, "gate", prefix="exp"),
                session_id=evidence.session_id,
                reward=sample_reward,
                baseline=sample_baseline,
                advantage=sample_reward - sample_baseline,
                success=promotion_safe,
            )
        ]
        replay = self._ppo_replay_surrogate(
            candidate,
            baseline=baseline,
            samples=sample_batch,
        )
        j_score = replay["j_score"]
        margin = 0.03
        parent_safe = baseline is None or not _learned_skill_unstable(baseline)
        trust_region_safe = j_score > -margin and (
            baseline is None or replay["candidate_alignment"] >= replay["baseline_alignment"] - margin
        )
        accepted = promotion_safe and parent_safe and (
            (
                evidence.success
                and baseline is None
                and candidate_score > 0
                and trust_region_safe
            ) or (
                candidate_score >= baseline_score + margin
                and trust_region_safe
            ) or (
                evidence.success
                and baseline is not None
                and current_reward >= baseline_score + margin
                and trust_region_safe
            )
        )
        reason = (
            "candidate passed Skill-Pro PPO gate"
            if accepted
            else (
                "candidate failed Skill-Pro PPO gate: unsafe partial outcome"
                if not promotion_safe
                else "candidate failed Skill-Pro PPO gate: unstable parent skill"
                if not parent_safe
                else "candidate failed Skill-Pro PPO gate"
            )
        )
        return PPOGateDecision(
            accepted=accepted,
            reason=reason,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            replaced_skill_id=baseline.skill_id if baseline and accepted else None,
            candidate_skill_id=candidate.skill_id,
            parent_skill_id=baseline.skill_id if baseline else "",
            j_score=j_score,
            candidate_alignment=replay["candidate_alignment"],
            baseline_alignment=replay["baseline_alignment"],
            sample_count=len(sample_batch),
            best_of_n=best_of_n,
            candidate_type="REFINE" if candidate_type == "REFINE" else "NEW",
        )

    def learn_from_episode(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> dict[str, Any]:
        state = self.store.load()
        maintenance_logs = self._normalize_experience_pools(state)
        if maintenance_logs:
            state.maintenance_log.extend(maintenance_logs)
        total_added_tokens = int(evidence.metrics.get("memory_total_added_tokens") or 0)
        delta_prompt_tokens_per_step = (
            total_added_tokens / max(evidence.steps or len(tool_steps), 1)
        )
        if not tool_steps:
            state_changed = bool(maintenance_logs)
            if not any(item.session_id == evidence.session_id for item in state.episodes):
                state.episodes.append(public_episode_evidence(evidence))
                state_changed = True
            if state_changed:
                self.store.save(state)
            return {
                "status": "rejected",
                "reason": "Skill-Pro requires at least one observed execution step.",
                "skill_id": "",
                "decision": None,
                "skills": len(state.skills),
                "total_added_tokens": total_added_tokens,
                "delta_prompt_tokens_per_step": round(
                    delta_prompt_tokens_per_step,
                    6,
                ),
                "prompt_added_tokens": int(
                    evidence.metrics.get("memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get("memory_tool_description_added_tokens") or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("memory_followup_added_tokens") or 0
                ),
            }

        parent = self._runtime_parent_from_steps(state, tool_steps)
        if parent is None:
            parent_item = self._select_parent_for_evidence(
                evidence=evidence,
                tool_steps=tool_steps,
            )
            parent = parent_item.skill if parent_item is not None else None
        if parent is not None and _learned_skill_unstable(parent):
            parent = None
        reward = _evidence_score(evidence)
        baseline_value = state.baselines.get(evidence.scenario or "default", 0.0)
        promotion_safe = _safe_skill_promotion(evidence.metrics)
        stat_reward, stat_baseline = _skill_stat_reward(
            evidence,
            reward,
            baseline_value,
        )
        replay_reward = reward if promotion_safe else stat_reward
        runtime_skill_counts = self._runtime_skill_counts(state, tool_steps)
        experience_skill_ids = [
            skill_id
            for skill_id, count in runtime_skill_counts.items()
            for _ in range(count)
        ]
        if not experience_skill_ids and parent is not None:
            experience_skill_ids = [parent.skill_id]
        experience = self._experience_from_episode(
            evidence=evidence,
            tool_steps=tool_steps,
            reward=replay_reward,
            baseline=stat_baseline,
            skill_ids=experience_skill_ids,
            success=promotion_safe,
        )

        if not any(item.session_id == evidence.session_id for item in state.episodes):
            state.episodes.append(public_episode_evidence(evidence))
        if not any(item.experience_id == experience.experience_id for item in state.experiences):
            state.experiences.append(experience)
            state.experiences = state.experiences[-EXPERIENCE_POOL_SIZE:]
        self._update_golden_pool(state, experience)
        self._update_baseline(
            state,
            evidence.scenario or "default",
            reward if promotion_safe else stat_reward,
        )
        state.iteration += 1
        for skill in state.skills.values():
            skill.increment_maturity()
        if runtime_skill_counts:
            total_calls = sum(runtime_skill_counts.values())
            for skill_id, count in runtime_skill_counts.items():
                state.skills[skill_id].update_stats(
                    reward=stat_reward,
                    baseline=stat_baseline,
                    total_skill_calls=total_calls,
                    skill_call_count=count,
                )
        elif parent is not None and parent.skill_id in state.skills:
            state.skills[parent.skill_id].update_stats(
                reward=stat_reward,
                baseline=stat_baseline,
                total_skill_calls=1,
                skill_call_count=1,
            )

        if not promotion_safe:
            reason = (
                "episode outcome is unsafe for Skill-Pro promotion: "
                "detection, localization, and RCA must all be sufficiently supported"
            )
            self._maintain(state)
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": parent.skill_id if parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "candidate": "",
                    "action": "rejected",
                    "reason": reason,
                    "sample_count": 0,
                    "required_sample_count": self.evolution_threshold,
                }
            )
            self.store.save(state)
            return {
                "status": "rejected",
                "reason": reason,
                "skill_id": parent.skill_id if parent else "",
                "episode_reward": reward,
                "episode_baseline": baseline_value,
                "episode_advantage": reward - baseline_value,
                "episode_success": promotion_safe,
                "total_added_tokens": experience.total_added_tokens,
                "delta_prompt_tokens_per_step": round(
                    experience.total_added_tokens
                    / max(evidence.steps or len(tool_steps), 1),
                    6,
                ),
                "prompt_added_tokens": int(
                    evidence.metrics.get("memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get("memory_tool_description_added_tokens") or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("memory_followup_added_tokens") or 0
                ),
                "prompt_injection_count": int(
                    evidence.metrics.get("memory_prompt_injection_count") or 0
                ),
                "tool_description_injection_count": int(
                    evidence.metrics.get("memory_tool_description_injection_count") or 0
                ),
                "followup_guidance_count": int(
                    evidence.metrics.get("memory_followup_guidance_count") or 0
                ),
                "semantic_gradient_source": "not_promoted",
                "semantic_gradient_llm_attempted": False,
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_llm_error": "",
                "decision": None,
                "sample_count": 0,
                "required_sample_count": self.evolution_threshold,
                "skills": len(state.skills),
                "experience_id": experience.experience_id,
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "method": "Skill-Pro",
            }

        samples = self._evolution_batch(state, parent, current=experience)
        if len(samples) < self.evolution_threshold:
            self._maintain(state)
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": parent.skill_id if parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "candidate": "",
                    "action": "deferred",
                    "reason": "insufficient Skill-Pro evolution batch",
                    "sample_count": len(samples),
                    "required_sample_count": self.evolution_threshold,
                }
            )
            self.store.save(state)
            return {
                "status": "deferred",
                "reason": "insufficient Skill-Pro evolution batch",
                "skill_id": parent.skill_id if parent else "",
                "episode_reward": reward,
                "episode_baseline": baseline_value,
                "episode_advantage": reward - baseline_value,
                "episode_success": promotion_safe,
                "total_added_tokens": experience.total_added_tokens,
                "delta_prompt_tokens_per_step": round(
                    experience.total_added_tokens
                    / max(evidence.steps or len(tool_steps), 1),
                    6,
                ),
                "prompt_added_tokens": int(
                    evidence.metrics.get("memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get("memory_tool_description_added_tokens") or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("memory_followup_added_tokens") or 0
                ),
                "prompt_injection_count": int(
                    evidence.metrics.get("memory_prompt_injection_count") or 0
                ),
                "tool_description_injection_count": int(
                    evidence.metrics.get("memory_tool_description_injection_count") or 0
                ),
                "followup_guidance_count": int(
                    evidence.metrics.get("memory_followup_guidance_count") or 0
                ),
                "semantic_gradient_source": "deferred",
                "semantic_gradient_llm_attempted": False,
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_llm_error": "",
                "decision": None,
                "sample_count": len(samples),
                "required_sample_count": self.evolution_threshold,
                "skills": len(state.skills),
                "experience_id": experience.experience_id,
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "method": "Skill-Pro",
            }

        episode_gradient = self.semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        best_decision: PPOGateDecision | None = None
        best_candidate: ProceduralSkill | None = None
        candidate_type = "REFINE" if parent is not None else "NEW"
        for index in range(self.best_of_n):
            candidate = self.propose_skill(
                evidence=evidence,
                tool_steps=tool_steps,
                parent=parent,
                candidate_index=index,
                critique=episode_gradient,
            )
            decision = self.ppo_gate(
                candidate=candidate,
                evidence=evidence,
                baseline=parent,
                samples=samples,
                candidate_type=candidate_type,
                best_of_n=self.best_of_n,
            )
            if best_decision is None or decision.j_score > best_decision.j_score:
                best_decision = decision
                best_candidate = candidate

        assert best_decision is not None and best_candidate is not None
        gradient_source = (
            best_candidate.semantic_gradients[-1].gradient_source
            if best_candidate.semantic_gradients
            else "deterministic"
        )
        gradient_error = (
            best_candidate.semantic_gradients[-1].llm_error
            if best_candidate.semantic_gradients
            else ""
        )
        if best_decision.accepted:
            old = state.skills.get(best_candidate.skill_id)
            if old is not None:
                best_candidate.reuse_count = old.reuse_count
                best_candidate.frequency = old.frequency
                best_candidate.total_gain = old.total_gain
                best_candidate.avg_gain = old.avg_gain
                best_candidate.success_count += old.success_count
                best_candidate.failure_count += old.failure_count
                best_candidate.source_sessions = sorted(set(old.source_sessions + best_candidate.source_sessions))
                best_candidate.semantic_gradients = old.semantic_gradients + best_candidate.semantic_gradients
            if parent is not None and parent.skill_id in state.skills:
                state.skills[parent.skill_id].last_evolved_iteration = state.iteration
            state.skills[best_candidate.skill_id] = best_candidate
        sample_ids = [sample.experience_id for sample in samples]
        for item in state.experiences:
            if item.experience_id in sample_ids:
                item.used_for_evolution = True
        for item in state.golden_experiences:
            if item.experience_id in sample_ids:
                item.used_for_evolution = True
        state.ppo_decisions.append(best_decision)
        state.evolution_log.append(
            {
                "iteration": state.iteration,
                "parent": parent.skill_id if parent else "",
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "sample_experience_ids": sample_ids,
                "candidate": best_candidate.skill_id,
                "action": "accepted" if best_decision.accepted else "rejected",
                "j_score": best_decision.j_score,
                "candidate_alignment": best_decision.candidate_alignment,
                "baseline_alignment": best_decision.baseline_alignment,
                "sample_count": best_decision.sample_count,
                "best_of_n": self.best_of_n,
                "semantic_gradient_source": gradient_source,
                "semantic_gradient_llm_error": gradient_error,
            }
        )
        self._maintain(state)
        self.store.save(state)
        return {
            "status": "accepted" if best_decision.accepted else "rejected",
            "skill_id": best_candidate.skill_id,
            "episode_reward": reward,
            "episode_baseline": baseline_value,
            "episode_advantage": reward - baseline_value,
            "episode_success": promotion_safe,
            "total_added_tokens": experience.total_added_tokens,
            "delta_prompt_tokens_per_step": round(
                experience.total_added_tokens / max(evidence.steps or len(tool_steps), 1),
                6,
            ),
            "prompt_added_tokens": int(
                evidence.metrics.get("memory_prompt_added_tokens") or 0
            ),
            "tool_description_added_tokens": int(
                evidence.metrics.get("memory_tool_description_added_tokens") or 0
            ),
            "followup_added_tokens": int(
                evidence.metrics.get("memory_followup_added_tokens") or 0
            ),
            "prompt_injection_count": int(
                evidence.metrics.get("memory_prompt_injection_count") or 0
            ),
            "tool_description_injection_count": int(
                evidence.metrics.get("memory_tool_description_injection_count") or 0
            ),
            "followup_guidance_count": int(
                evidence.metrics.get("memory_followup_guidance_count") or 0
            ),
            "semantic_gradient_source": gradient_source,
            "semantic_gradient_llm_attempted": bool(self.llm_backend and self.model),
            "semantic_gradient_llm_failed": bool(gradient_error),
            "semantic_gradient_llm_error": gradient_error,
            "decision": best_decision.model_dump(),
            "skills": len(state.skills),
            "experience_id": experience.experience_id,
            "runtime_skill_ids": sorted(set(experience_skill_ids)),
            "method": "Skill-Pro",
        }

    def _runtime_skill_counts(
        self,
        state,
        tool_steps: list[SkillStep],
    ) -> Counter[str]:
        return Counter(
            step.skill_id
            for step in tool_steps
            if step.skill_id and step.skill_id in state.skills
        )

    def _runtime_parent_from_steps(
        self,
        state,
        tool_steps: list[SkillStep],
    ) -> ProceduralSkill | None:
        counts = self._runtime_skill_counts(state, tool_steps)
        if not counts:
            return None
        skill_id, _ = counts.most_common(1)[0]
        skill = state.skills.get(skill_id)
        if skill is not None and _learned_skill_unstable(skill):
            return None
        return skill

    def _select_parent_for_evidence(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SkillRetrieval | None:
        attrs = infer_memory_attributes(
            _episode_attribute_text(evidence, tool_steps),
            scenario=evidence.scenario,
            topology_class=evidence.topology_class,
            tools=[step.tool_name for step in tool_steps if step.tool_name],
        )
        return self.select_skill(
            query=MemoryQuery(
                text=evidence.task_description,
                scenario=evidence.scenario,
                topology_class=evidence.topology_class,
                protocols=attrs.protocols,
                services=attrs.services,
                symptoms=attrs.symptoms,
                tools=attrs.tools,
                top_k=3,
            ),
            record_reuse=False,
        )

    def _experience_from_episode(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        reward: float,
        baseline: float,
        skill_ids: list[str],
        success: bool,
    ) -> SkillExperience:
        transitions = [
            SkillTransition(
                state=evidence.task_description,
                action=step.action,
                skill_id=step.skill_id or (skill_ids[0] if skill_ids else ""),
                tool_name=step.tool_name,
                arguments_hint=step.arguments_hint,
                observation_summary=step.observation_summary,
                status=step.status,
                done=index == len(tool_steps) - 1,
            )
            for index, step in enumerate(tool_steps)
        ]
        return SkillExperience(
            experience_id=_stable_id(evidence.session_id, [step.model_dump(mode="json") for step in tool_steps], prefix="exp"),
            session_id=evidence.session_id,
            reward=reward,
            baseline=baseline,
            advantage=reward - baseline,
            skill_ids=skill_ids,
            trajectory=evidence.task_description,
            scenario=evidence.scenario,
            transitions=transitions,
            step_count=evidence.steps,
            total_added_tokens=int(evidence.metrics.get("memory_total_added_tokens") or 0),
            success=success,
        )

    def _evolution_batch(
        self,
        state,
        parent: ProceduralSkill | None,
        *,
        current: SkillExperience | None = None,
    ) -> list[SkillExperience]:
        if parent is None:
            pool = [
                exp for exp in state.experiences if not exp.used_for_evolution
            ][-self.evolution_threshold :]
        else:
            pool = [
                exp
                for exp in state.experiences
                if parent.skill_id in exp.skill_ids
                and not exp.used_for_evolution
            ]
            if len(pool) < self.evolution_threshold:
                pool = pool + [
                    exp
                    for exp in state.golden_experiences
                    if exp.experience_id not in {item.experience_id for item in pool}
                    and not exp.used_for_evolution
                ]
        if current is not None and (parent is None or _uses_generic_seed_policy(parent)):
            current_signature = _experience_signature(current)
            if any(current_signature):
                clustered = [
                    exp
                    for exp in pool
                    if exp.experience_id == current.experience_id
                    or _compatible_experience_signature(
                        current_signature,
                        _experience_signature(exp),
                    )
                ]
                pool = clustered
        if len(pool) <= self.evolution_threshold:
            return list(pool)
        ordered = sorted(pool, key=lambda exp: exp.reward)
        low_count = max(1, self.evolution_threshold // 2)
        high_count = max(1, self.evolution_threshold - low_count)
        batch = ordered[:low_count] + ordered[-high_count:]
        seen: dict[str, SkillExperience] = {}
        for exp in batch:
            seen[exp.experience_id] = exp
        return list(seen.values())

    def _transition_alignment(
        self,
        skill: ProceduralSkill | None,
        experience: SkillExperience,
    ) -> float:
        """Score how well a Skill-MDP option explains replayed transitions.

        Skill-Pro's reference implementation verifies candidates by replaying
        saved experience.  We do not have token log-probs for NIKA traces, so
        this deterministic proxy compares the candidate's initiation/policy/
        termination text with the actual tool/action/observation sequence.
        """
        if skill is None or not experience.transitions:
            return 0.0
        skill_text = " ".join(
            [
                skill.title,
                skill.activation_condition,
                " ".join(step.action for step in skill.execution_steps),
                skill.termination_condition,
                " ".join(skill.tools),
                " ".join(skill.protocols),
                " ".join(skill.services),
                " ".join(skill.symptoms),
            ]
        )
        step_texts = [step.action for step in skill.execution_steps]
        scores: list[float] = []
        for transition in experience.transitions:
            transition_text = " ".join(
                [
                    transition.action,
                    transition.tool_name,
                    _dump_for_alignment(transition.arguments_hint),
                    transition.observation_summary,
                    transition.status,
                ]
            )
            lexical = _jaccard(skill_text, transition_text)
            policy = max(
                [_jaccard(step_text, transition_text) for step_text in step_texts]
                or [0.0]
            )
            tool = 0.0
            if transition.tool_name:
                if transition.tool_name in skill.tools:
                    tool = 1.0
                elif transition.tool_name.lower() in skill_text.lower():
                    tool = 0.75
                elif any(
                    token in _tokens(skill_text)
                    for token in _tokens(transition.tool_name)
                ):
                    tool = 0.45
            termination = 0.0
            if transition.done:
                termination = _jaccard(
                    skill.termination_condition,
                    " ".join([transition.action, transition.observation_summary]),
                )
            status_weight = 0.85 if transition.status == "error" else 1.0
            scores.append(
                status_weight
                * _clamp(
                    (0.35 * lexical)
                    + (0.35 * policy)
                    + (0.2 * tool)
                    + (0.1 * termination),
                    0.0,
                    1.0,
                )
            )
        return sum(scores) / max(len(scores), 1)

    def _ppo_replay_surrogate(
        self,
        candidate: ProceduralSkill,
        *,
        baseline: ProceduralSkill | None,
        samples: list[SkillExperience],
    ) -> dict[str, float]:
        if not samples:
            candidate_score = self._skill_effective_score(candidate)
            baseline_score = self._skill_effective_score(baseline) if baseline else 0.0
            return {
                "j_score": candidate_score - baseline_score,
                "candidate_alignment": candidate_score,
                "baseline_alignment": baseline_score,
            }
        total = 0.0
        steps = 0
        candidate_alignment_total = 0.0
        baseline_alignment_total = 0.0
        for exp in samples:
            candidate_alignment = self._transition_alignment(candidate, exp)
            baseline_alignment = (
                self._transition_alignment(baseline, exp) if baseline else 0.0
            )
            alignment_delta = candidate_alignment - baseline_alignment
            raw_ratio = math.exp(_clamp(alignment_delta, -2.0, 2.0))
            clipped_ratio = _clamp(
                raw_ratio,
                1.0 - self.ppo_epsilon,
                1.0 + self.ppo_epsilon,
            )
            advantage = (
                exp.advantage
                if exp.advantage != 0
                else exp.reward - exp.baseline
            )
            transition_count = max(len(exp.transitions), 1)
            per_step = advantage / transition_count
            surrogate = min(raw_ratio * per_step, clipped_ratio * per_step)
            total += surrogate * transition_count
            steps += transition_count
            candidate_alignment_total += candidate_alignment * transition_count
            baseline_alignment_total += baseline_alignment * transition_count
        return {
            "j_score": total / max(steps, 1),
            "candidate_alignment": candidate_alignment_total / max(steps, 1),
            "baseline_alignment": baseline_alignment_total / max(steps, 1),
        }

    def _ppo_surrogate(
        self,
        candidate: ProceduralSkill,
        *,
        baseline: ProceduralSkill | None,
        samples: list[SkillExperience],
    ) -> float:
        return self._ppo_replay_surrogate(
            candidate,
            baseline=baseline,
            samples=samples,
        )["j_score"]

    def _update_baseline(self, state, scenario: str, reward: float) -> None:
        old = state.baselines.get(scenario)
        state.baselines[scenario] = reward if old is None else (1 - BASELINE_EMA_ALPHA) * old + BASELINE_EMA_ALPHA * reward

    def _update_golden_pool(self, state, experience: SkillExperience) -> None:
        if not experience.transitions or not experience.success:
            return
        pool = {item.experience_id: item for item in state.golden_experiences}
        pool[experience.experience_id] = experience
        state.golden_experiences = sorted(pool.values(), key=lambda item: item.reward, reverse=True)[:GOLDEN_POOL_SIZE]

    def _lcb(self, skill: ProceduralSkill) -> float:
        if skill.frequency <= 0 and (skill.success_count + skill.failure_count) <= 0:
            return 0.0
        state = self.store.load()
        t = max([item.maturity for item in state.skills.values()] or [1])
        n = max(skill.frequency, 1)
        return skill.avg_gain - 0.2 * math.sqrt(math.log1p(t) / n)

    def _lcb_bonus(self, skill: ProceduralSkill, total_maturity: int) -> float:
        n = max(skill.frequency, 1)
        return max(-0.2, skill.avg_gain - 0.2 * math.sqrt(math.log1p(max(total_maturity, 1)) / n))

    def _skill_lcb_from_current_state(self, skill: ProceduralSkill) -> float:
        return self._skill_lcb_from_state(skill, self.store.load())

    def _skill_effective_score(self, skill: ProceduralSkill) -> float:
        observed = skill.success_count + skill.failure_count
        base = max(skill.score, skill.avg_gain)
        if observed <= 0:
            return max(0.0, base)
        success_rate = skill.success_count / observed
        confidence = min(1.0, observed / 5.0)
        reliability_cap = ((1.0 - confidence) * base) + (confidence * success_rate)
        return max(0.0, min(base, reliability_cap))

    def _normalize_experience_pools(self, state) -> list[dict[str, Any]]:
        episodes_by_session = {
            episode.session_id: episode
            for episode in state.episodes
            if episode.session_id and episode.metrics
        }
        logs: list[dict[str, Any]] = []
        repaired_ids: set[str] = set()

        def repair(experience: SkillExperience) -> None:
            evidence = episodes_by_session.get(experience.session_id)
            if evidence is None:
                return
            safe = _safe_skill_promotion(evidence.metrics)
            changed = False
            if experience.success != safe:
                experience.success = safe
                changed = True
            if not safe:
                raw_reward = _evidence_score(evidence)
                repaired_reward, _ = _skill_stat_reward(
                    evidence,
                    raw_reward,
                    experience.baseline,
                )
                if experience.reward > repaired_reward:
                    experience.reward = repaired_reward
                    experience.advantage = experience.reward - experience.baseline
                    changed = True
            if changed and experience.experience_id not in repaired_ids:
                repaired_ids.add(experience.experience_id)
                logs.append(
                    {
                        "stage": "normalize unsafe experience",
                        "experience_id": experience.experience_id,
                        "session_id": experience.session_id,
                    }
                )

        for experience in state.experiences:
            repair(experience)
        for experience in state.golden_experiences:
            repair(experience)

        experiences_by_id = {
            experience.experience_id: experience for experience in state.experiences
        }
        kept: dict[str, SkillExperience] = {}
        removed_ids: list[str] = []
        for experience in state.golden_experiences:
            normalized = experiences_by_id.get(experience.experience_id, experience)
            if normalized.transitions and normalized.success:
                kept[normalized.experience_id] = normalized
            else:
                removed_ids.append(normalized.experience_id)
        if removed_ids:
            logs.append(
                {
                    "stage": "remove unsafe golden experience",
                    "experience_ids": sorted(set(removed_ids)),
                }
            )
        state.golden_experiences = sorted(
            kept.values(),
            key=lambda item: item.reward,
            reverse=True,
        )[:GOLDEN_POOL_SIZE]
        return logs

    def _maintain(self, state) -> None:
        logs = self._normalize_experience_pools(state)
        seen_hashes: dict[str, str] = {}
        active = [skill for skill in state.skills.values() if skill.status != "retired"]
        for skill in active:
            digest = skill.content_hash()
            duplicate_of = seen_hashes.get(digest)
            if duplicate_of and skill.maturity >= 3:
                skill.status = "retired"
                logs.append({"stage": "duplicate skill", "skill_id": skill.skill_id, "duplicate_of": duplicate_of})
            else:
                seen_hashes[digest] = skill.skill_id
            if skill.frequency >= 3 and self._skill_lcb_from_state(skill, state) < -0.05:
                skill.status = "retired"
                logs.append({"stage": "negative LCB", "skill_id": skill.skill_id})
            if _learned_skill_unstable(skill):
                skill.status = "retired"
                logs.append({"stage": "unstable learned skill", "skill_id": skill.skill_id})
            skill.score = self._skill_effective_score(skill)
        active = [skill for skill in state.skills.values() if skill.status != "retired"]
        if len(active) > self.pool_size:
            ranked = sorted(
                active,
                key=lambda item: (self._skill_lcb_from_state(item, state), item.score, -item.maturity),
                reverse=True,
            )
            keep_ids = {skill.skill_id for skill in ranked[: self.pool_size]}
            for skill in active:
                if skill.skill_id not in keep_ids:
                    skill.status = "retired"
                    logs.append({"stage": "capacity overflow", "skill_id": skill.skill_id})
        if logs:
            state.maintenance_log.extend(logs)

    def _skill_lcb_from_state(self, skill: ProceduralSkill, state) -> float:
        t = max([item.maturity for item in state.skills.values()] or [1])
        n = max(skill.frequency, 1)
        return skill.avg_gain - 0.2 * math.sqrt(math.log1p(t) / n)
