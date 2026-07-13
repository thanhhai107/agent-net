"""Skill-Pro Procedural Memory service.

This adapts the official Skill-Pro semantics to NIKA's diagnosis-agent
boundary: a Skill-MDP style selector injects active procedural skills before
diagnosis, while closed benchmark episodes feed an ExperiencePool /
GoldenExperiencePool and non-parametric PPO-style evolution gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
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
from agent.extensions.llm import load_extension_model as load_model
from agent.procedural_memory.safety import redact_oracle_markers
from agent.procedural_memory.attributes import infer_procedural_memory_attributes
from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralMemoryQuery,
    PPOGateDecision,
    ProceduralSkill,
    SemanticGradient,
    SemanticGradientDraft,
    SkillCandidateDraft,
    SkillComponentGradient,
    SkillExperience,
    SkillRetrieval,
    SkillStep,
    SkillTransition,
    utc_now,
)
from agent.procedural_memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyLogprobScorer,
    PolicyScorer,
    StructuredReplayPolicyScorer,
)
from agent.procedural_memory.policy_context import serialize_primitive_action
from agent.procedural_memory.store import ProceduralMemoryStore, public_episode_evidence

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
SEED_SKILL_IDS = GENERIC_SEED_SKILL_IDS
BASELINE_EMA_ALPHA = 0.1
SELECTION_EPSILON_DECAY_EPISODES = 500


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _metric_success(
    metrics: dict[str, Any],
    ground_truth_is_anomaly: bool | None = None,
) -> bool:
    detection_complete = float(metrics.get("detection_score") or 0) >= 1.0
    if ground_truth_is_anomaly is False:
        return detection_complete
    return (
        detection_complete
        and _component_complete(metrics, "localization")
        and _component_complete(metrics, "rca")
    )


def _safe_skill_promotion(
    metrics: dict[str, Any],
    ground_truth_is_anomaly: bool | None = None,
) -> bool:
    return _metric_success(metrics, ground_truth_is_anomaly)


def _component_complete(metrics: dict[str, Any], prefix: str) -> bool:
    accuracy = float(metrics.get(f"{prefix}_accuracy") or 0.0)
    f1 = float(metrics.get(f"{prefix}_f1") or 0.0)
    return max(accuracy, f1) >= 1.0


def _component_reward(metrics: dict[str, Any], prefix: str) -> float:
    for suffix in ("f1", "accuracy", "precision"):
        value = metrics.get(f"{prefix}_{suffix}")
        if value is not None:
            return _clamp(float(value), 0.0, 1.0)
    return 0.0


def _evidence_score(evidence: EvaluationEvidence) -> float:
    detection = float(evidence.metrics.get("detection_score") or 0.0)
    localization = _component_reward(evidence.metrics, "localization")
    rca = _component_reward(evidence.metrics, "rca")
    is_anomaly = evidence.ground_truth_is_anomaly
    if is_anomaly is None and (evidence.root_cause or evidence.faulty_devices):
        is_anomaly = True
    if is_anomaly is False:
        quality = detection
    else:
        quality = (detection + localization + rca) / 3.0
    return _clamp(quality, 0.0, 1.0)


def _skill_stat_reward(
    evidence: EvaluationEvidence, reward: float, baseline: float
) -> tuple[float, float]:
    """Return the continuous reward/baseline used by online score maintenance."""

    del evidence
    return reward, baseline


def _baseline_key(evidence: EvaluationEvidence) -> str:
    return evidence.scenario or "default"


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
    return "network diagnosis"


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
    return skill.frequency >= 10 and skill.avg_gain <= 0.0


def _activation_similarity(
    skill: ProceduralSkill, query: ProceduralMemoryQuery
) -> float:
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
    scope_hits = (
        len(query_tokens & scope_tokens) / max(len(scope_tokens), 1)
        if scope_tokens
        else 0.0
    )
    discriminators = {
        token
        for token in activation_tokens
        if len(token) >= 5
        and token not in {"current", "evidence", "diagnosis", "symptoms"}
    }
    discriminator_hits = (
        len(query_tokens & discriminators) / max(len(discriminators), 1)
        if discriminators
        else 0.0
    )
    return _clamp(
        (0.55 * lexical) + (0.3 * scope_hits) + (0.15 * discriminator_hits), 0.0, 1.0
    )


def _signature_activation(signature: str) -> str:
    return (
        "Use when the current observation history matches this "
        f"evidence signature: {signature}. Do not activate from "
        "scenario name or tool catalog alone."
    )


def _experience_signature(
    exp: SkillExperience,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
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
    attrs = infer_procedural_memory_attributes(text, scenario=exp.scenario, tools=tools)
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
    comparable = [
        len(left & right) / len(left | right)
        for left, right in (
            (left_protocols, right_protocols),
            (left_services, right_services),
            (left_symptoms, right_symptoms),
        )
        if left and right
    ]
    if not comparable:
        return False
    return sum(comparable) / len(comparable) >= 0.6


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
        evolution_threshold: int = 6,
        best_of_n: int = 3,
        ppo_epsilon: float = PPO_EPSILON,
        policy_scorer: PolicyScorer | None = None,
    ) -> None:
        self.bank_id = bank_id
        self.llm_backend = llm_backend
        self.model = model
        self.pool_size = pool_size
        self.evolution_threshold = evolution_threshold
        self.best_of_n = max(1, best_of_n)
        self.ppo_epsilon = ppo_epsilon
        self._learning_llm_instance: Any | None = None
        self.policy_scorer = policy_scorer or self._default_policy_scorer()
        self.store = ProceduralMemoryStore(
            bank_id=bank_id,
            root=store_path.parent if store_path else None,
        )
        self._ensure_seed_skills()

    def _default_policy_scorer(self) -> PolicyScorer:
        selected_backend = learning_backend(self.llm_backend)
        selected_model = learning_model(self.model)
        if selected_backend == "custom" and selected_model:
            base_url = (
                os.getenv("NIKA_SKILL_LOGPROB_URL", "").strip()
                or os.getenv("CUSTOM_API_URL", "").strip()
            )
            api_key = (
                os.getenv("NIKA_SKILL_LOGPROB_API_KEY", "").strip()
                or os.getenv("CUSTOM_API_KEY", "").strip()
            )
            scorer_model = (
                os.getenv("NIKA_SKILL_LOGPROB_MODEL", "").strip() or selected_model
            )
            if base_url and api_key:
                return PolicyLogprobScorer(
                    base_url=base_url,
                    api_key=api_key,
                    model=scorer_model,
                    timeout=learning_timeout_seconds(),
                )
        if selected_backend and selected_model:
            return BehavioralReplayPolicyScorer(self._learning_llm)
        return StructuredReplayPolicyScorer()

    def _learning_llm(self) -> Any | None:
        selected_backend = learning_backend(self.llm_backend)
        selected_model = learning_model(self.model)
        if not selected_backend or not selected_model:
            return None
        if self._learning_llm_instance is None:
            self._learning_llm_instance = load_model(
                selected_backend,
                selected_model,
                timeout=learning_timeout_seconds(),
                max_retries=learning_max_retries(),
            )
        return self._learning_llm_instance

    def clear(self) -> None:
        self.store.clear()
        self._ensure_seed_skills()

    def _ensure_seed_skills(self) -> None:
        with self.store.exclusive():
            state = self.store.load()
            changed = False
            for skill in self._seed_skills():
                if skill.skill_id not in state.skills:
                    state.skills[skill.skill_id] = skill
                    changed = True
                else:
                    stored = state.skills[skill.skill_id]
                    if stored.origin != skill.origin:
                        stored.origin = skill.origin
                        changed = True
                    if stored.prior_score == 0.0:
                        stored.prior_score = skill.prior_score or skill.score
                        changed = True
                    if stored.status != "validated":
                        stored.status = "validated"
                        changed = True
            legacy_expert_skill_ids = [
                skill_id
                for skill_id, skill in state.skills.items()
                if skill.origin == "expert_seed"
            ]
            for skill_id in legacy_expert_skill_ids:
                del state.skills[skill_id]
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
                prior_score=0.25,
                origin="generic_seed",
            )
            for skill_id, title, activation, policy, termination in seed_specs
        ]
        return skills

    def retrieve(
        self, *, query: ProceduralMemoryQuery, session_id: str = ""
    ) -> list[SkillRetrieval]:
        state = self.store.load()
        scored: list[SkillRetrieval] = []
        query_text = " ".join(
            [
                query.text,
                " ".join(query.protocols),
                " ".join(query.services),
                " ".join(query.symptoms),
                " ".join(query.tools),
            ]
        ).lower()
        for skill in state.skills.values():
            if skill.status != "validated":
                continue
            if _learned_skill_unstable(skill):
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
            if (
                not _is_seed_skill(skill)
                and activation_fit < 0.03
                and (skill.protocols or skill.services or skill.symptoms)
            ):
                continue
            score += 0.35 * activation_fit
            if activation_fit > 0:
                reasons.append(f"activation_fit:{activation_fit:.2f}")
            for label, values in (
                ("protocol", skill.protocols),
                ("service", skill.services),
                ("symptom", skill.symptoms),
                ("tool", skill.tools),
            ):
                query_values = (
                    getattr(query, f"{label}s", []) if label != "tool" else query.tools
                )
                overlap = set(values) & set(query_values)
                if overlap:
                    score += 0.15 * len(overlap)
                    reasons.append(f"{label}:{','.join(sorted(overlap))}")
                elif query_values and values:
                    score -= 0.12 if label in {"protocol", "tool"} else 0.06
            for token in skill.activation_condition.lower().split():
                if len(token) > 3 and token in query_text:
                    score += 0.01
            if score > 0:
                scored.append(SkillRetrieval(skill=skill, score=score, reasons=reasons))
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
        query: ProceduralMemoryQuery,
        session_id: str = "",
        top_k: int | None = None,
        record_reuse: bool = True,
        exclude_skill_ids: Collection[str] | None = None,
        allow_excluded_fallback: bool = True,
        exploration_epsilon: float = 0.0,
        exploration_key: str = "",
    ) -> SkillRetrieval | None:
        top_k = top_k or query.top_k
        state = self.store.load()
        excluded = {skill_id for skill_id in (exclude_skill_ids or []) if skill_id}
        candidates = self.retrieve(
            query=query.model_copy(update={"top_k": max(top_k + len(excluded), 1)}),
            session_id=session_id,
        )
        selectable = [
            item for item in candidates if item.skill.skill_id not in excluded
        ]
        pool = selectable or (candidates if allow_excluded_fallback else [])
        exploration_triggered = self._exploration_triggered(
            exploration_epsilon, exploration_key
        )
        selected = self._exploration_choice(
            state=state,
            candidates=pool,
            epsilon=exploration_epsilon,
            key=exploration_key,
            excluded_skill_ids=excluded,
        )
        if selected is None and not exploration_triggered and pool:
            selected = max(
                pool[: max(top_k, 1)],
                key=lambda item: (self._skill_effective_score(item.skill), item.score),
            )
        if selected is None:
            return None
        if record_reuse:
            selected = self._record_skill_reuse(selected)
        return selected

    @staticmethod
    def _stable_unit_interval(value: str) -> float:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _exploration_triggered(self, epsilon: float, key: str) -> bool:
        return bool(
            epsilon > 0 and key and self._stable_unit_interval(f"{key}:gate") < epsilon
        )

    def _exploration_choice(
        self,
        *,
        state,
        candidates: list[SkillRetrieval],
        epsilon: float,
        key: str,
        excluded_skill_ids: set[str] | None = None,
    ) -> SkillRetrieval | None:
        if not self._exploration_triggered(epsilon, key):
            return None
        excluded = {item.skill.skill_id for item in candidates} | set(
            excluded_skill_ids or ()
        )
        alternatives = [
            SkillRetrieval(
                skill=skill,
                score=self._skill_effective_score(skill),
                reasons=["epsilon_exploration"],
            )
            for skill in state.skills.values()
            if skill.status == "validated"
            and not _learned_skill_unstable(skill)
            and skill.skill_id not in excluded
        ]
        choices: list[SkillRetrieval | None] = [*candidates, *alternatives, None]
        index = int(self._stable_unit_interval(f"{key}:choice") * len(choices))
        return choices[min(index, len(choices) - 1)]

    def decayed_selection_epsilon(self, initial: float) -> float:
        iteration = self.store.load().iteration
        progress = min(
            max(iteration, 0) / float(SELECTION_EPSILON_DECAY_EPISODES),
            1.0,
        )
        return initial + progress * (0.05 - initial)

    def _transfer_scope_adjustment(
        self,
        *,
        skill: ProceduralSkill,
        query: ProceduralMemoryQuery,
    ) -> tuple[float, list[str], bool]:
        if _is_seed_skill(skill):
            return 0.0, [], False
        reasons: list[str] = []
        delta = 0.0
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
        if (
            discriminating_skill_labels
            and discriminating_query_labels
            and overlap_count == 0
        ):
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

    def _record_skill_reuse(self, selected: SkillRetrieval) -> SkillRetrieval:
        with self.store.exclusive():
            state = self.store.load()
            stored = state.skills.get(selected.skill.skill_id)
            if stored is not None:
                stored.reuse_count += 1
                stored.updated_at = utc_now()
                state.skills[stored.skill_id] = stored
                self.store.save(state)
                selected.skill = stored
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
        output_path.write_text(
            "\n".join(self.store.snapshot_jsonl()) + "\n", encoding="utf-8"
        )
        return output_path

    def propose_skill(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        parent: ProceduralSkill | None = None,
        candidate_index: int = 0,
        critique: SemanticGradient | None = None,
        experiences: list[SkillExperience] | None = None,
    ) -> ProceduralSkill:
        if not tool_steps:
            raise ValueError("Skill-Pro requires at least one observed execution step.")
        attrs = infer_procedural_memory_attributes(
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
        sampled = self._llm_skill_candidate(
            evidence=evidence,
            parent=parent,
            critique=critique,
            experiences=experiences or [],
            candidate_index=candidate_index,
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
            steps = [
                SkillStep(
                    order=index + 1,
                    action=step.action,
                    tool_name=step.tool_name,
                    rationale="Distilled from an observed Skill-Pro transition.",
                )
                for index, step in enumerate(tool_steps[:10])
            ]
            termination = (
                "Stop when current observations support the anomaly decision; if an "
                "anomaly is present, also require supported localization and root cause. "
                "Do not invent localization or root cause for a no-anomaly conclusion."
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
                SkillStep(
                    order=i + 1, action=step, rationale="Skill-Pro semantic update."
                )
                for i, step in enumerate(critique.component_update.policy)
                if step.strip()
            ]
            steps = update_steps or parent.execution_steps
            termination = (
                critique.component_update.termination or parent.termination_condition
            )
            parent_id = parent.skill_id
        if sampled is not None:
            title = sampled.title.strip() or title
            activation = sampled.initiation.strip() or activation
            sampled_steps = [
                SkillStep(
                    order=index + 1,
                    action=action.strip(),
                    rationale="Independently sampled Skill-Pro candidate.",
                )
                for index, action in enumerate(sampled.policy[:8])
                if action.strip()
            ]
            if sampled_steps:
                steps = sampled_steps
            termination = sampled.termination.strip() or termination
        if critique.proposed_update:
            termination += f" Semantic update: {critique.proposed_update[:240]}"
        if sampled is None and candidate_index == 1:
            termination += (
                " Require independent confirmation before the final conclusion."
            )
        elif sampled is None and candidate_index >= 2:
            steps = steps + [
                SkillStep(
                    order=len(steps) + 1,
                    action="Cross-check the leading hypothesis with an independent tool before submitting.",
                    rationale="Best-of-N Skill-Pro candidate variant.",
                )
            ]
        outcome_success = _safe_skill_promotion(
            evidence.metrics,
            evidence.ground_truth_is_anomaly,
        )
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
            prior_score=_evidence_score(evidence),
            parent_id=parent_id,
            version=version,
            semantic_gradients=[critique],
            origin="learned",
        )

    def _llm_skill_candidate(
        self,
        *,
        evidence: EvaluationEvidence,
        parent: ProceduralSkill | None,
        critique: SemanticGradient,
        experiences: list[SkillExperience],
        candidate_index: int,
    ) -> SkillCandidateDraft | None:
        """Sample one candidate independently from the aggregated batch signal."""
        if not learning_backend(self.llm_backend) or not learning_model(self.model):
            return None
        parent_payload = (
            parent.format_for_llm()
            if parent is not None
            else "No parent skill. Create a new reusable procedural skill."
        )
        batch_payload = [
            {
                "session_id": item.session_id,
                "reward": item.reward,
                "baseline": item.baseline,
                "advantage": item.advantage,
                "success": item.success,
                "metrics": item.metrics,
                "transitions": [
                    {
                        "tool": transition.tool_name,
                        "status": transition.status,
                        "observation": _trim_text(
                            transition.observation_summary,
                            limit=180,
                        ),
                    }
                    for transition in item.transitions[:6]
                ],
            }
            for item in experiences[-6:]
        ]
        observed_tools = sorted(
            {
                transition.tool_name
                for item in experiences
                for transition in item.transitions
                if transition.tool_name
            }
        )
        prompt = (
            "You are the Skill-Pro Skill Evolver. Produce one independently "
            "sampled procedural candidate from a batch-aggregated semantic "
            "gradient. Generalize across trajectories; do not copy device names, "
            "hidden labels, benchmark case ids, or one-off output values.\n\n"
            "Every concrete tool, output field, protocol layer, threshold, and "
            "termination test must be supported by the batch below or by the "
            "parent skill. Do not invent canonical probes, fixed numeric updates, "
            "topology models, or tool capabilities. Reasoning steps may stay "
            "abstract when the evidence does not support a concrete operation. "
            "The skill owns action ordering, evidence interpretation, initiation, "
            "and termination. Tool Refinement owns parameter schemas, preconditions, "
            "constraints, failure modes, and return contracts; do not copy or rewrite "
            "those details into the skill.\n"
            f"Observed tool names: {json.dumps(observed_tools, ensure_ascii=False)}\n\n"
            f"Candidate sample index: {candidate_index}\n"
            f"Parent skill:\n{parent_payload}\n\n"
            "Aggregated semantic gradient:\n"
            f"{critique.model_dump_json(indent=2)}\n\n"
            "Batch evidence summaries:\n"
            f"{json.dumps(batch_payload, indent=2, ensure_ascii=False)}\n\n"
            "Return SkillCandidateDraft with a concise title, observable "
            "initiation condition, 3-6 executable policy steps, and a checkable "
            "termination condition. Make this sample behaviorally distinct from "
            "other candidate indices while preserving the aggregated evidence."
        )
        try:
            llm = self._learning_llm()
            if llm is None:
                return None
            evolver = llm.with_structured_output(SkillCandidateDraft)
            raw = evolver.invoke(prompt)
            draft = (
                raw
                if isinstance(raw, SkillCandidateDraft)
                else SkillCandidateDraft.model_validate(raw)
            )
            draft.title = _redact_hidden_labels(
                redact_oracle_markers(draft.title), evidence
            )
            draft.initiation = _redact_hidden_labels(
                redact_oracle_markers(draft.initiation), evidence
            )
            draft.policy = [
                _redact_hidden_labels(redact_oracle_markers(step), evidence)
                for step in draft.policy
            ]
            draft.termination = _redact_hidden_labels(
                redact_oracle_markers(draft.termination), evidence
            )
            if not draft.policy:
                return None
            return draft
        except Exception:
            return None

    def semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        skill: ProceduralSkill | None = None,
    ) -> SemanticGradient:
        llm_gradient, llm_error = self._llm_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
            skill=skill,
        )
        if llm_gradient is not None:
            return llm_gradient
        gradient = self._deterministic_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        gradient.llm_error = llm_error
        return gradient

    def semantic_gradient_from_experience(
        self,
        *,
        experience: SkillExperience,
        skill: ProceduralSkill | None,
    ) -> SemanticGradient:
        """Recover one per-trajectory semantic gradient from replay state."""
        steps = [
            SkillStep(
                order=index + 1,
                action=transition.action,
                skill_id=transition.skill_id,
                tool_name=transition.tool_name,
                arguments_hint=transition.arguments_hint,
                observation_summary=transition.observation_summary,
                status=transition.status,
                rationale="Replayed Skill-Pro trajectory transition.",
            )
            for index, transition in enumerate(experience.transitions)
        ]
        evidence = EvaluationEvidence(
            session_id=experience.session_id,
            task_description=experience.trajectory,
            scenario=experience.scenario,
            metrics=experience.metrics,
            ground_truth_is_anomaly=experience.ground_truth_is_anomaly,
            steps=experience.step_count or len(steps),
            tool_calls=len([step for step in steps if step.tool_name]),
            success=experience.success,
        )
        return self.semantic_gradient(
            evidence=evidence,
            tool_steps=steps,
            skill=skill,
        )

    def aggregate_semantic_gradients(
        self,
        *,
        gradients: list[SemanticGradient],
        skill: ProceduralSkill | None,
        source_session_id: str,
        evidence: EvaluationEvidence | None = None,
    ) -> SemanticGradient:
        """Consolidate per-trajectory gradients into one stable batch update."""
        if not gradients:
            return SemanticGradient(
                source_session_id=source_session_id,
                critique="No related trajectory gradients were available.",
                proposed_update="Preserve the current skill until more evidence arrives.",
                component_update=SkillComponentGradient(is_related=False),
            )
        if learning_backend(self.llm_backend) and learning_model(self.model):
            prompt = (
                "You are the Skill-Pro batch semantic-gradient aggregator. "
                "Consolidate recurring, causally supported updates and discard "
                "conflicting or trajectory-specific details. Do not include device "
                "names, hidden labels, benchmark ids, or exact one-off values.\n\n"
                "Parent skill:\n"
                f"{skill.format_for_llm() if skill else 'NEW SKILL'}\n\n"
                "Per-trajectory gradients:\n"
                f"{json.dumps([item.model_dump(mode='json') for item in gradients], indent=2, ensure_ascii=False)}\n\n"
                "Return one SemanticGradientDraft. Keep policy to 3-6 short, "
                "executable steps. Set is_related=false only if no recurring "
                "signal applies to the parent skill."
            )
            try:
                llm = self._learning_llm()
                if llm is None:
                    raise RuntimeError("learning LLM is unavailable")
                aggregator = llm.with_structured_output(SemanticGradientDraft)
                raw = aggregator.invoke(prompt)
                draft = (
                    raw
                    if isinstance(raw, SemanticGradientDraft)
                    else SemanticGradientDraft.model_validate(raw)
                )
                gradient = SemanticGradient(
                    source_session_id=source_session_id,
                    critique=redact_oracle_markers(draft.critique),
                    proposed_update=redact_oracle_markers(draft.proposed_update),
                    component_update=SkillComponentGradient(
                        initiation=redact_oracle_markers(draft.initiation),
                        policy=[
                            redact_oracle_markers(step)
                            for step in draft.policy[:6]
                            if step.strip()
                        ],
                        termination=redact_oracle_markers(draft.termination),
                        is_related=draft.is_related,
                    ),
                    gradient_source="llm",
                )
                if evidence is not None:
                    gradient.critique = _redact_hidden_labels(
                        gradient.critique, evidence
                    )
                    gradient.proposed_update = _redact_hidden_labels(
                        gradient.proposed_update, evidence
                    )
                    gradient.component_update.initiation = _redact_hidden_labels(
                        gradient.component_update.initiation, evidence
                    )
                    gradient.component_update.policy = [
                        _redact_hidden_labels(step, evidence)
                        for step in gradient.component_update.policy
                    ]
                    gradient.component_update.termination = _redact_hidden_labels(
                        gradient.component_update.termination, evidence
                    )
                return gradient
            except Exception as exc:
                aggregate_error = format_learning_error(exc)
        else:
            aggregate_error = ""

        related = [item for item in gradients if item.component_update.is_related]
        selected = related or gradients
        policies: list[str] = []
        for item in selected:
            for step in item.component_update.policy:
                if step and step not in policies:
                    policies.append(step)
        initiations = [
            item.component_update.initiation
            for item in selected
            if item.component_update.initiation
        ]
        terminations = [
            item.component_update.termination
            for item in selected
            if item.component_update.termination
        ]
        return SemanticGradient(
            source_session_id=source_session_id,
            critique="Batch patterns: "
            + " | ".join(item.critique for item in selected[:4] if item.critique),
            proposed_update=" | ".join(
                item.proposed_update for item in selected[:4] if item.proposed_update
            ),
            component_update=SkillComponentGradient(
                initiation=initiations[0] if initiations else "",
                policy=policies[:6],
                termination=terminations[0] if terminations else "",
                is_related=bool(related),
            ),
            gradient_source="deterministic",
            llm_error=aggregate_error,
        )

    def _deterministic_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        if _safe_skill_promotion(
            evidence.metrics,
            evidence.ground_truth_is_anomaly,
        ):
            critique = (
                "Successful trajectory: preserve evidence order and termination rule."
            )
            update = "Promote or reinforce this procedure if it improves step/tool efficiency."
            termination = (
                "Terminate after current evidence supports a no-anomaly conclusion; "
                "leave localization and root cause empty."
                if evidence.ground_truth_is_anomaly is False
                else "Terminate after direct evidence supports detection, localization, and RCA."
            )
            component = SkillComponentGradient(
                policy=[step.action for step in tool_steps[:6]],
                termination=termination,
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
                    "Summarize observations relevant to the skill's activation condition.",
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
        skill: ProceduralSkill | None = None,
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
            "success": evidence.success,
            "reward": _evidence_score(evidence),
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
            f"Skill being evaluated:\n{skill.format_for_llm() if skill else 'NEW SKILL'}\n\n"
            f"Observed execution steps:\n{json.dumps(_skill_steps_summary(tool_steps), indent=2, ensure_ascii=False)}\n\n"
            "Return a compact SemanticGradientDraft with critique, proposed_update, "
            "initiation, policy, termination, and is_related. Keep policy to at most "
            "four short steps. Keep tool parameter schemas, preconditions, constraints, "
            "failure modes, and return contracts in Tool Refinement rather than the "
            "procedural skill. Use the same source_session_id."
        )
        try:
            llm = self._learning_llm()
            if llm is None:
                return None, "learning LLM is unavailable"
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
            gradient.critique = _redact_hidden_labels(
                redact_oracle_markers(gradient.critique), evidence
            )
            gradient.proposed_update = _redact_hidden_labels(
                redact_oracle_markers(gradient.proposed_update), evidence
            )
            gradient.component_update.initiation = _redact_hidden_labels(
                redact_oracle_markers(gradient.component_update.initiation),
                evidence,
            )
            gradient.component_update.policy = [
                _redact_hidden_labels(redact_oracle_markers(step), evidence)
                for step in gradient.component_update.policy
            ]
            gradient.component_update.termination = _redact_hidden_labels(
                redact_oracle_markers(gradient.component_update.termination),
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
                success=_safe_skill_promotion(
                    evidence.metrics,
                    evidence.ground_truth_is_anomaly,
                ),
            )
        ]
        replay = self._ppo_replay_surrogate(
            candidate,
            baseline=baseline,
            samples=sample_batch,
        )
        j_score = replay["j_score"]
        margin = 0.001
        parent_safe = baseline is None or not _learned_skill_unstable(baseline)
        verified_success_count = sum(item.success for item in sample_batch)
        positive_advantage_count = sum(item.advantage > 0 for item in sample_batch)
        verification_error = str(replay.get("verification_error") or "")
        accepted = parent_safe and not verification_error and j_score > margin
        reason = (
            "candidate passed Skill-Pro PPO gate"
            if accepted
            else "candidate failed Skill-Pro PPO gate: verification unavailable"
            if verification_error
            else "candidate failed Skill-Pro PPO gate: unstable parent skill"
            if not parent_safe
            else "candidate failed Skill-Pro PPO gate"
        )
        return PPOGateDecision(
            accepted=accepted,
            reason=reason,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            replaced_skill_id=None,
            candidate_skill_id=candidate.skill_id,
            parent_skill_id=baseline.skill_id if baseline else "",
            j_score=j_score,
            candidate_alignment=replay["candidate_alignment"],
            baseline_alignment=replay["baseline_alignment"],
            sample_count=len(sample_batch),
            best_of_n=best_of_n,
            candidate_type="REFINE" if candidate_type == "REFINE" else "NEW",
            verification_method=str(
                replay.get("verification_method") or "structured_replay"
            ),
            verified_success_count=verified_success_count,
            positive_advantage_count=positive_advantage_count,
            verification_error=verification_error,
        )

    def learn_from_episode(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> dict[str, Any]:
        with self.store.exclusive():
            return self._learn_from_episode_unlocked(
                evidence=evidence,
                tool_steps=tool_steps,
            )

    def _learn_from_episode_unlocked(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> dict[str, Any]:
        state = self.store.load()
        maintenance_logs = self._normalize_experience_pools(state)
        if maintenance_logs:
            state.maintenance_log.extend(maintenance_logs)
        total_added_tokens = int(
            evidence.metrics.get("procedural_memory_total_added_tokens") or 0
        )
        delta_prompt_tokens_per_step = total_added_tokens / max(
            evidence.steps or len(tool_steps), 1
        )
        if not tool_steps:
            state_changed = bool(maintenance_logs)
            if not any(
                item.session_id == evidence.session_id for item in state.episodes
            ):
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
                    evidence.metrics.get("procedural_memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get(
                        "procedural_memory_tool_description_added_tokens"
                    )
                    or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("procedural_memory_followup_added_tokens") or 0
                ),
            }

        parent = self._runtime_parent_from_steps(state, tool_steps)
        if parent is not None and _learned_skill_unstable(parent):
            parent = None
        reward = _evidence_score(evidence)
        baseline_key = _baseline_key(evidence)
        baseline_value = state.baselines.get(baseline_key, 0.0)
        promotion_safe = _safe_skill_promotion(
            evidence.metrics,
            evidence.ground_truth_is_anomaly,
        )
        stat_reward, stat_baseline = _skill_stat_reward(
            evidence,
            reward,
            baseline_value,
        )
        runtime_skill_counts = self._runtime_skill_counts(state, tool_steps)
        experience_skill_ids = sorted(runtime_skill_counts)
        attributed_steps = sum(
            step.skill_id in runtime_skill_counts for step in tool_steps
        )
        unattributed_steps = max(0, len(tool_steps) - attributed_steps)
        attribution_rate = attributed_steps / max(len(tool_steps), 1)
        segment_experiences = self._segment_experiences(
            evidence=evidence,
            tool_steps=tool_steps,
            reward=stat_reward,
            baseline=stat_baseline,
            success=promotion_safe,
            valid_skill_ids=set(runtime_skill_counts),
        )
        target_segment_id = parent.skill_id if parent is not None else ""
        experience = segment_experiences.get(target_segment_id)
        if experience is None:
            self._maintain(state)
            self.store.save(state)
            return {
                "status": "rejected",
                "reason": "No execution segment was attributable to the selected parent.",
                "skill_id": "",
                "episode_reward": reward,
                "episode_baseline": baseline_value,
                "episode_advantage": reward - baseline_value,
                "attributed_steps": attributed_steps,
                "unattributed_steps": unattributed_steps,
                "attribution_rate": round(attribution_rate, 6),
            }
        target_tool_steps = [
            step
            for step in tool_steps
            if (step.skill_id if step.skill_id in runtime_skill_counts else "")
            == target_segment_id
        ]

        if not any(item.session_id == evidence.session_id for item in state.episodes):
            state.episodes.append(public_episode_evidence(evidence))
        existing_experience_ids = {item.experience_id for item in state.experiences}
        current_experience_ids = sorted(
            segment.experience_id for segment in segment_experiences.values()
        )
        for segment in segment_experiences.values():
            if segment.experience_id not in existing_experience_ids:
                state.experiences.append(segment)
                existing_experience_ids.add(segment.experience_id)
            self._update_golden_pool(state, segment)
        state.experiences = state.experiences[-EXPERIENCE_POOL_SIZE:]
        self._update_baseline(
            state,
            baseline_key,
            stat_reward,
        )
        state.iteration += 1
        for skill in state.skills.values():
            if skill.status != "retired":
                skill.increment_maturity()
        if runtime_skill_counts:
            total_calls = max(sum(runtime_skill_counts.values()), 1)
            for skill_id, count in runtime_skill_counts.items():
                state.skills[skill_id].update_stats(
                    reward=stat_reward,
                    baseline=stat_baseline,
                    total_skill_calls=total_calls,
                    skill_call_count=count,
                )

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
                    "attributed_steps": attributed_steps,
                    "unattributed_steps": unattributed_steps,
                    "attribution_rate": round(attribution_rate, 6),
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
                    evidence.metrics.get("procedural_memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get(
                        "procedural_memory_tool_description_added_tokens"
                    )
                    or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("procedural_memory_followup_added_tokens") or 0
                ),
                "prompt_injection_count": int(
                    evidence.metrics.get("procedural_memory_prompt_injection_count")
                    or 0
                ),
                "tool_description_injection_count": int(
                    evidence.metrics.get(
                        "procedural_memory_tool_description_injection_count"
                    )
                    or 0
                ),
                "followup_guidance_count": int(
                    evidence.metrics.get("procedural_memory_followup_guidance_count")
                    or 0
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
                "experience_ids": current_experience_ids,
                "attributed_steps": attributed_steps,
                "unattributed_steps": unattributed_steps,
                "attribution_rate": round(attribution_rate, 6),
                "method": "Skill-Pro",
            }

        per_trajectory_gradients: list[SemanticGradient] = []
        for sample in samples:
            if sample.experience_id == experience.experience_id:
                gradient = self.semantic_gradient(
                    evidence=evidence,
                    tool_steps=target_tool_steps,
                    skill=parent,
                )
            else:
                gradient = self.semantic_gradient_from_experience(
                    experience=sample,
                    skill=parent,
                )
            gradient.critique = _redact_hidden_labels(
                redact_oracle_markers(gradient.critique), evidence
            )
            gradient.proposed_update = _redact_hidden_labels(
                redact_oracle_markers(gradient.proposed_update), evidence
            )
            gradient.component_update.initiation = _redact_hidden_labels(
                redact_oracle_markers(gradient.component_update.initiation), evidence
            )
            gradient.component_update.policy = [
                _redact_hidden_labels(redact_oracle_markers(step), evidence)
                for step in gradient.component_update.policy
            ]
            gradient.component_update.termination = _redact_hidden_labels(
                redact_oracle_markers(gradient.component_update.termination), evidence
            )
            per_trajectory_gradients.append(gradient)
        related_gradient_count = sum(
            gradient.component_update.is_related
            for gradient in per_trajectory_gradients
        )
        related_gradients = [
            gradient
            for gradient in per_trajectory_gradients
            if gradient.component_update.is_related
        ]
        batch_gradient = self.aggregate_semantic_gradients(
            gradients=related_gradients,
            skill=parent,
            source_session_id=evidence.session_id,
            evidence=evidence,
        )
        verification_samples = self._verification_batch(
            state,
            parent,
            generation_samples=samples,
            current=experience,
        )
        if not verification_samples:
            reason = "insufficient disjoint Skill-Pro verification batch"
            self._maintain(state)
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": parent.skill_id if parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "sample_experience_ids": [
                        sample.experience_id for sample in samples
                    ],
                    "verification_experience_ids": [],
                    "candidate": "",
                    "action": "deferred",
                    "reason": reason,
                    "sample_count": len(samples),
                    "required_sample_count": self.evolution_threshold,
                    "required_verification_count": 1,
                    "attributed_steps": attributed_steps,
                    "unattributed_steps": unattributed_steps,
                    "attribution_rate": round(attribution_rate, 6),
                }
            )
            self.store.save(state)
            return {
                "status": "deferred",
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
                    evidence.metrics.get("procedural_memory_prompt_added_tokens") or 0
                ),
                "tool_description_added_tokens": int(
                    evidence.metrics.get(
                        "procedural_memory_tool_description_added_tokens"
                    )
                    or 0
                ),
                "followup_added_tokens": int(
                    evidence.metrics.get("procedural_memory_followup_added_tokens") or 0
                ),
                "semantic_gradient_source": "deferred",
                "semantic_gradient_llm_attempted": False,
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_llm_error": "",
                "semantic_gradient_count": 0,
                "verification_method": "none",
                "verification_error": "",
                "decision": None,
                "sample_count": len(samples),
                "required_sample_count": self.evolution_threshold,
                "verification_sample_count": 0,
                "required_verification_count": 1,
                "skills": len(state.skills),
                "experience_id": experience.experience_id,
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "experience_ids": current_experience_ids,
                "attributed_steps": attributed_steps,
                "unattributed_steps": unattributed_steps,
                "attribution_rate": round(attribution_rate, 6),
                "method": "Skill-Pro",
            }
        best_decision: PPOGateDecision | None = None
        best_candidate: ProceduralSkill | None = None
        relevance_ratio = related_gradient_count / max(len(per_trajectory_gradients), 1)
        candidate_type = (
            "REFINE" if parent is not None and relevance_ratio >= 0.5 else "NEW"
        )
        for index in range(self.best_of_n):
            candidate = self.propose_skill(
                evidence=evidence,
                tool_steps=target_tool_steps,
                parent=parent if candidate_type == "REFINE" else None,
                candidate_index=index,
                critique=batch_gradient,
                experiences=samples,
            )
            candidate.status = "candidate"
            candidate.success_count = 0
            candidate.failure_count = 0
            candidate.score = (
                self._skill_effective_score(parent) if parent is not None else 0.0
            )
            candidate.prior_score = candidate.score
            decision = self.ppo_gate(
                candidate=candidate,
                evidence=evidence,
                baseline=parent,
                samples=verification_samples,
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
            best_candidate.status = "validated"
            best_candidate.source_sessions = sorted(
                set(best_candidate.source_sessions)
                | {
                    sample.session_id
                    for sample in verification_samples
                    if sample.success and sample.session_id
                }
            )
            if parent is not None and candidate_type == "REFINE":
                best_candidate.source_sessions = sorted(
                    set(parent.source_sessions + best_candidate.source_sessions)
                )
                best_candidate.semantic_gradients = (
                    parent.semantic_gradients + best_candidate.semantic_gradients
                )
            old = state.skills.get(best_candidate.skill_id)
            if old is not None:
                best_candidate.reuse_count = old.reuse_count
                best_candidate.frequency = old.frequency
                best_candidate.total_gain = old.total_gain
                best_candidate.avg_gain = old.avg_gain
                best_candidate.success_count += old.success_count
                best_candidate.failure_count += old.failure_count
                best_candidate.source_sessions = sorted(
                    set(old.source_sessions + best_candidate.source_sessions)
                )
                best_candidate.semantic_gradients = (
                    old.semantic_gradients + best_candidate.semantic_gradients
                )
            if (
                parent is not None
                and candidate_type == "REFINE"
                and parent.skill_id in state.skills
            ):
                state.skills[parent.skill_id].last_evolved_iteration = state.iteration
            state.skills[best_candidate.skill_id] = best_candidate
        sample_ids = [sample.experience_id for sample in samples]
        retained_ids: set[str] = set()
        if not best_decision.accepted:
            retain_count = max(1, self.evolution_threshold // 2)
            retained_ids = {sample.experience_id for sample in samples[-retain_count:]}
        for item in state.experiences:
            if (
                item.experience_id in sample_ids
                and item.experience_id not in retained_ids
            ):
                item.used_for_evolution = True
        state.ppo_decisions.append(best_decision)
        state.evolution_log.append(
            {
                "iteration": state.iteration,
                "parent": parent.skill_id if parent else "",
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "sample_experience_ids": sample_ids,
                "verification_experience_ids": [
                    sample.experience_id for sample in verification_samples
                ],
                "candidate": best_candidate.skill_id,
                "action": "accepted" if best_decision.accepted else "rejected",
                "j_score": best_decision.j_score,
                "candidate_alignment": best_decision.candidate_alignment,
                "baseline_alignment": best_decision.baseline_alignment,
                "sample_count": best_decision.sample_count,
                "best_of_n": self.best_of_n,
                "semantic_gradient_source": gradient_source,
                "semantic_gradient_llm_error": gradient_error,
                "semantic_gradient_count": len(per_trajectory_gradients),
                "related_semantic_gradient_count": related_gradient_count,
                "relevance_ratio": round(relevance_ratio, 6),
                "candidate_type": candidate_type,
                "verification_method": best_decision.verification_method,
                "verification_error": best_decision.verification_error,
                "attributed_steps": attributed_steps,
                "unattributed_steps": unattributed_steps,
                "attribution_rate": round(attribution_rate, 6),
            }
        )
        self._maintain(state)
        self.store.save(state)
        return {
            "status": "accepted" if best_decision.accepted else "rejected",
            "reason": best_decision.reason,
            "skill_id": best_candidate.skill_id,
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
                evidence.metrics.get("procedural_memory_prompt_added_tokens") or 0
            ),
            "tool_description_added_tokens": int(
                evidence.metrics.get("procedural_memory_tool_description_added_tokens")
                or 0
            ),
            "followup_added_tokens": int(
                evidence.metrics.get("procedural_memory_followup_added_tokens") or 0
            ),
            "prompt_injection_count": int(
                evidence.metrics.get("procedural_memory_prompt_injection_count") or 0
            ),
            "tool_description_injection_count": int(
                evidence.metrics.get(
                    "procedural_memory_tool_description_injection_count"
                )
                or 0
            ),
            "followup_guidance_count": int(
                evidence.metrics.get("procedural_memory_followup_guidance_count") or 0
            ),
            "semantic_gradient_source": gradient_source,
            "semantic_gradient_llm_attempted": bool(
                learning_backend(self.llm_backend) and learning_model(self.model)
            ),
            "semantic_gradient_llm_failed": bool(gradient_error),
            "semantic_gradient_llm_error": gradient_error,
            "semantic_gradient_count": len(per_trajectory_gradients),
            "related_semantic_gradient_count": related_gradient_count,
            "relevance_ratio": round(relevance_ratio, 6),
            "candidate_type": candidate_type,
            "verification_method": best_decision.verification_method,
            "verification_error": best_decision.verification_error,
            "decision": best_decision.model_dump(),
            "skills": len(state.skills),
            "experience_id": experience.experience_id,
            "runtime_skill_ids": sorted(set(experience_skill_ids)),
            "experience_ids": current_experience_ids,
            "attributed_steps": attributed_steps,
            "unattributed_steps": unattributed_steps,
            "attribution_rate": round(attribution_rate, 6),
            "method": "Skill-Pro",
        }

    def _runtime_skill_counts(
        self,
        state,
        tool_steps: list[SkillStep],
    ) -> Counter[str]:
        valid_steps = [
            step
            for step in tool_steps
            if step.skill_id
            and step.skill_id in state.skills
            and state.skills[step.skill_id].status != "retired"
            and not _learned_skill_unstable(state.skills[step.skill_id])
        ]
        activations: set[tuple[str, str]] = set()
        fallback_runs: Counter[str] = Counter()
        previous_skill_id = ""
        for step in valid_steps:
            if step.activation_id:
                activations.add((step.skill_id, step.activation_id))
            elif step.skill_id != previous_skill_id:
                fallback_runs[step.skill_id] += 1
            previous_skill_id = step.skill_id
        counts = Counter(skill_id for skill_id, _ in activations)
        counts.update(fallback_runs)
        return counts

    def _runtime_parent_from_steps(
        self,
        state,
        tool_steps: list[SkillStep],
    ) -> ProceduralSkill | None:
        counts = self._runtime_skill_counts(state, tool_steps)
        if not counts:
            return None
        unused_counts = Counter(
            skill_id
            for experience in state.experiences
            if not experience.used_for_evolution
            for skill_id in experience.skill_ids
            if skill_id in counts
        )
        ready_ids = [
            skill_id
            for skill_id in counts
            if unused_counts[skill_id] + 1 >= self.evolution_threshold
        ]
        if ready_ids:
            skill_id = max(
                ready_ids,
                key=lambda item: (
                    self._skill_evolution_priority(state, state.skills[item]),
                    unused_counts[item],
                    counts[item],
                    item,
                ),
            )
        else:
            skill_id, _ = counts.most_common(1)[0]
        skill = state.skills.get(skill_id)
        if skill is not None and _learned_skill_unstable(skill):
            return None
        return skill

    def _skill_evolution_priority(self, state, skill: ProceduralSkill) -> float:
        if (
            skill.last_evolved_iteration > 0
            and state.iteration - skill.last_evolved_iteration < 3
        ):
            return -1.0
        peers = [
            item
            for item in state.skills.values()
            if item.status != "retired" and item.frequency > 0
        ]
        reference_gain = max((item.avg_gain for item in peers), default=0.0)
        gap = max(0.0, reference_gain - skill.avg_gain)
        frequency = max(skill.frequency, 0)
        impact = 1.0 - math.exp(-frequency / 50.0)
        confidence = 1.0 - math.exp(-frequency / 30.0)
        age_factor = math.exp(-max(skill.maturity, 0) / 8.0)
        return impact * confidence * gap * age_factor

    def _experience_from_episode(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        reward: float,
        baseline: float,
        skill_ids: list[str],
        success: bool,
        transitions: list[SkillTransition] | None = None,
    ) -> SkillExperience:
        transitions = transitions or self._transitions_from_steps(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        return SkillExperience(
            experience_id=_stable_id(
                evidence.session_id,
                skill_ids,
                [step.model_dump(mode="json") for step in tool_steps],
                prefix="exp",
            ),
            session_id=evidence.session_id,
            reward=reward,
            baseline=baseline,
            advantage=reward - baseline,
            skill_ids=skill_ids,
            trajectory=evidence.task_description,
            scenario=evidence.scenario,
            metrics={
                str(key): float(value)
                for key, value in evidence.metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            transitions=transitions,
            step_count=len(transitions),
            total_added_tokens=int(
                evidence.metrics.get("procedural_memory_total_added_tokens") or 0
            ),
            success=success,
            ground_truth_is_anomaly=evidence.ground_truth_is_anomaly,
        )

    def _transitions_from_steps(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> list[SkillTransition]:
        transitions: list[SkillTransition] = []
        observation_history: list[str] = []
        for index, step in enumerate(tool_steps):
            state = evidence.task_description
            if observation_history:
                state += "\nRecent observations:\n" + "\n".join(
                    observation_history[-4:]
                )
            action = step.action
            if step.tool_name:
                action = serialize_primitive_action(
                    step.tool_name,
                    step.arguments_hint,
                )
            transitions.append(
                SkillTransition(
                    state=step.policy_state or state,
                    action=action,
                    skill_id=step.skill_id,
                    tool_name=step.tool_name,
                    arguments_hint=step.arguments_hint,
                    observation_summary=step.observation_summary,
                    status=step.status,
                    done=index == len(tool_steps) - 1,
                    policy_context=step.policy_context,
                    activation_id=step.activation_id,
                )
            )
            if step.observation_summary:
                observation_history.append(
                    f"{step.tool_name or 'action'}: "
                    f"{_trim_text(step.observation_summary, limit=500)}"
                )
        return transitions

    def _segment_experiences(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        reward: float,
        baseline: float,
        success: bool,
        valid_skill_ids: set[str] | None = None,
    ) -> dict[str, SkillExperience]:
        """Build one replay experience per actually active Skill-MDP option."""

        all_transitions = self._transitions_from_steps(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        grouped: dict[str, tuple[list[SkillStep], list[SkillTransition]]] = {}
        for step, transition in zip(tool_steps, all_transitions, strict=True):
            skill_id = step.skill_id
            if valid_skill_ids is not None and skill_id not in valid_skill_ids:
                skill_id = ""
            grouped_steps, grouped_transitions = grouped.setdefault(
                skill_id,
                ([], []),
            )
            grouped_steps.append(
                step
                if skill_id == step.skill_id
                else step.model_copy(update={"skill_id": ""})
            )
            grouped_transitions.append(
                transition
                if skill_id == transition.skill_id
                else transition.model_copy(update={"skill_id": ""})
            )
        for _, grouped_transitions in grouped.values():
            for transition in grouped_transitions:
                transition.done = False
            grouped_transitions[-1].done = True
        return {
            skill_id: self._experience_from_episode(
                evidence=evidence,
                tool_steps=steps,
                reward=reward,
                baseline=baseline,
                skill_ids=[skill_id] if skill_id else [],
                success=success,
                transitions=transitions,
            )
            for skill_id, (steps, transitions) in grouped.items()
        }

    def _evolution_batch(
        self,
        state,
        parent: ProceduralSkill | None,
        *,
        current: SkillExperience | None = None,
    ) -> list[SkillExperience]:
        if parent is None:
            pool = [
                exp
                for exp in state.experiences
                if not exp.skill_ids and not exp.used_for_evolution
            ][-self.evolution_threshold :]
        else:
            pool = [
                exp
                for exp in state.experiences
                if parent.skill_id in exp.skill_ids and not exp.used_for_evolution
            ]
        if current is not None and parent is None:
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
        low_count = self.evolution_threshold // 2
        high_count = self.evolution_threshold - low_count
        batch = ordered[:low_count] + ordered[-high_count:]
        seen: dict[str, SkillExperience] = {}
        for exp in batch:
            seen[exp.experience_id] = exp
        return list(seen.values())

    def _verification_batch(
        self,
        state,
        parent: ProceduralSkill | None,
        *,
        generation_samples: list[SkillExperience],
        current: SkillExperience | None = None,
    ) -> list[SkillExperience]:
        """Use the behavior-policy batch used by Skill-Pro candidate generation."""
        del parent, current
        rebased: list[SkillExperience] = []
        for sample in generation_samples:
            baseline = state.baselines.get(
                sample.scenario or "default", sample.baseline
            )
            rebased.append(
                sample.model_copy(
                    update={
                        "baseline": baseline,
                        "advantage": sample.reward - baseline,
                    }
                )
            )
        return rebased

    def _ppo_replay_surrogate(
        self,
        candidate: ProceduralSkill,
        *,
        baseline: ProceduralSkill | None,
        samples: list[SkillExperience],
    ) -> dict[str, Any]:
        if not samples:
            candidate_score = self._skill_effective_score(candidate)
            baseline_score = self._skill_effective_score(baseline) if baseline else 0.0
            return {
                "j_score": candidate_score - baseline_score,
                "candidate_alignment": candidate_score,
                "baseline_alignment": baseline_score,
                "verification_method": "structured_replay",
                "verification_error": "",
            }
        replay_scores = self.policy_scorer.score_batch(
            candidate=candidate,
            baseline=baseline,
            experiences=samples,
        )
        if replay_scores.method == "policy_logprob":
            return self._logprob_replay_surrogate(
                replay_scores=replay_scores,
                samples=samples,
            )
        score_by_id = {item.experience_id: item for item in replay_scores.scores}
        total = 0.0
        steps = 0
        candidate_alignment_total = 0.0
        baseline_alignment_total = 0.0
        for exp in samples:
            score = score_by_id.get(exp.experience_id)
            if score is None:
                continue
            candidate_alignment = score.candidate_alignment
            baseline_alignment = score.baseline_alignment
            alignment_delta = candidate_alignment - baseline_alignment
            raw_ratio = math.exp(_clamp(alignment_delta, -2.0, 2.0))
            clipped_ratio = _clamp(
                raw_ratio,
                1.0 - self.ppo_epsilon,
                1.0 + self.ppo_epsilon,
            )
            advantage = (
                exp.advantage if exp.advantage != 0 else exp.reward - exp.baseline
            )
            transition_count = max(len(exp.transitions), 1)
            per_step = advantage / max(exp.step_count, transition_count, 1)
            surrogate = min(raw_ratio * per_step, clipped_ratio * per_step)
            total += surrogate * transition_count
            steps += transition_count
            candidate_alignment_total += candidate_alignment * transition_count
            baseline_alignment_total += baseline_alignment * transition_count
        return {
            "j_score": total / max(steps, 1),
            "candidate_alignment": candidate_alignment_total / max(steps, 1),
            "baseline_alignment": baseline_alignment_total / max(steps, 1),
            "verification_method": replay_scores.method,
            "verification_error": replay_scores.error,
        }

    def _logprob_replay_surrogate(
        self,
        *,
        replay_scores,
        samples: list[SkillExperience],
    ) -> dict[str, Any]:
        score_by_step = {
            (item.experience_id, item.transition_index): item
            for item in replay_scores.step_logprobs
        }
        total = 0.0
        steps = 0
        candidate_logprob_total = 0.0
        baseline_logprob_total = 0.0
        verification_error = replay_scores.error
        for experience in samples:
            transition_count = len(experience.transitions)
            if transition_count <= 0:
                continue
            advantage = (
                experience.advantage
                if experience.advantage != 0
                else experience.reward - experience.baseline
            )
            step_advantage = advantage / transition_count
            for index in range(transition_count):
                score = score_by_step.get((experience.experience_id, index))
                if score is None:
                    verification_error = (
                        verification_error
                        or "logprob scorer omitted a historical transition"
                    )
                    continue
                log_ratio = _clamp(
                    score.candidate_logprob - score.baseline_logprob,
                    -10.0,
                    10.0,
                )
                ratio = math.exp(log_ratio)
                clipped_ratio = _clamp(
                    ratio,
                    1.0 - self.ppo_epsilon,
                    1.0 + self.ppo_epsilon,
                )
                total += min(
                    ratio * step_advantage,
                    clipped_ratio * step_advantage,
                )
                candidate_logprob_total += score.candidate_logprob
                baseline_logprob_total += score.baseline_logprob
                steps += 1
        if steps <= 0:
            verification_error = (
                verification_error or "logprob scorer returned no steps"
            )
        return {
            "j_score": total / max(steps, 1),
            "candidate_alignment": candidate_logprob_total / max(steps, 1),
            "baseline_alignment": baseline_logprob_total / max(steps, 1),
            "verification_method": "policy_logprob",
            "verification_error": verification_error,
        }

    def _update_baseline(self, state, scenario: str, reward: float) -> None:
        old = state.baselines.get(scenario)
        state.baselines[scenario] = (
            reward
            if old is None
            else (1 - BASELINE_EMA_ALPHA) * old + BASELINE_EMA_ALPHA * reward
        )

    def _update_golden_pool(self, state, experience: SkillExperience) -> None:
        if not experience.transitions:
            return
        pool = {item.experience_id: item for item in state.golden_experiences}
        pool[experience.experience_id] = experience.model_copy(deep=True)
        state.golden_experiences = sorted(
            pool.values(), key=lambda item: item.reward, reverse=True
        )[:GOLDEN_POOL_SIZE]

    def _skill_effective_score(self, skill: ProceduralSkill) -> float:
        prior = skill.prior_score or max(skill.score, 0.0)
        if skill.frequency <= 0:
            return prior
        prior_weight = 2.0 if _is_seed_skill(skill) else 1.0
        return (prior_weight * prior + skill.total_gain) / (
            prior_weight + skill.frequency
        )

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
            safe = _safe_skill_promotion(
                evidence.metrics,
                evidence.ground_truth_is_anomaly,
            )
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
            if experience.used_for_evolution:
                experience.used_for_evolution = False

        experiences_by_id = {
            experience.experience_id: experience for experience in state.experiences
        }
        kept: dict[str, SkillExperience] = {}
        removed_ids: list[str] = []
        for experience in state.golden_experiences:
            normalized = experiences_by_id.get(experience.experience_id, experience)
            if normalized.transitions:
                golden = normalized.model_copy(deep=True)
                golden.used_for_evolution = False
                kept[golden.experience_id] = golden
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
        active = [skill for skill in state.skills.values() if skill.status != "retired"]
        current_iteration = max(
            [state.iteration, *(skill.maturity for skill in active)], default=1
        )

        def lcb(skill: ProceduralSkill) -> float:
            frequency = max(skill.frequency, 1)
            return skill.avg_gain - 0.2 * math.sqrt(
                math.log1p(max(current_iteration, 1)) / frequency
            )

        def retire(
            skill: ProceduralSkill,
            stage: str,
            *,
            duplicate_of: str = "",
        ) -> None:
            if skill.status == "retired":
                return
            skill.status = "retired"
            payload = {"stage": stage, "skill_id": skill.skill_id}
            if duplicate_of:
                payload["duplicate_of"] = duplicate_of
            logs.append(payload)

        for skill in active:
            skill.score = self._skill_effective_score(skill)
            if _learned_skill_unstable(skill):
                retire(skill, "non-positive online score")

        active = [skill for skill in active if skill.status != "retired"]

        by_content: dict[str, list[ProceduralSkill]] = defaultdict(list)
        for skill in active:
            by_content[skill.content_hash()].append(skill)
        for duplicates in by_content.values():
            if len(duplicates) < 2:
                continue
            keeper = max(
                duplicates,
                key=lambda item: (lcb(item), item.frequency, -item.maturity),
            )
            for duplicate in duplicates:
                if duplicate.skill_id != keeper.skill_id and duplicate.maturity >= 3:
                    retire(duplicate, "duplicate skill", duplicate_of=keeper.skill_id)

        active = [skill for skill in active if skill.status != "retired"]
        semantic_ranked = sorted(
            active,
            key=lambda item: (lcb(item), item.frequency, -item.maturity),
            reverse=True,
        )
        semantic_keepers: list[ProceduralSkill] = []
        for skill in semantic_ranked:
            duplicate = next(
                (
                    keeper
                    for keeper in semantic_keepers
                    if self._semantic_skill_similarity(skill, keeper) >= 0.90
                ),
                None,
            )
            if duplicate is not None and skill.maturity >= 3:
                retire(skill, "semantic duplicate", duplicate_of=duplicate.skill_id)
            else:
                semantic_keepers.append(skill)

        active = [skill for skill in active if skill.status != "retired"]
        by_base: dict[str, list[ProceduralSkill]] = defaultdict(list)
        for skill in active:
            if skill.maturity >= 4:
                by_base[_skill_base_id(skill.skill_id)].append(skill)
        for versions in by_base.values():
            if len(versions) < 2:
                continue
            keeper = max(versions, key=lambda item: (lcb(item), item.version))
            for version in versions:
                if (
                    version.skill_id != keeper.skill_id
                    and len([item for item in active if item.status != "retired"])
                    > self.pool_size
                ):
                    retire(version, "outdated variant", duplicate_of=keeper.skill_id)

        active = [skill for skill in active if skill.status != "retired"]
        while len(active) > self.pool_size:
            harmful = [
                skill
                for skill in active
                if skill.maturity > 2 and skill.frequency >= 10 and lcb(skill) < -0.001
            ]
            if not harmful:
                break
            retire(min(harmful, key=lcb), "negative LCB")
            active = [skill for skill in active if skill.status != "retired"]

        if len(active) > self.pool_size:
            veterans = [skill for skill in active if skill.frequency >= 80]
            middle = [skill for skill in active if 10 <= skill.frequency < 80]
            newcomers = [skill for skill in active if skill.frequency < 10]
            for tier in (veterans, middle, newcomers):
                tier.sort(key=lcb, reverse=True)
            keep_ids = {
                skill.skill_id
                for skill in (veterans + middle + newcomers)[: self.pool_size]
            }
            for skill in active:
                if skill.skill_id not in keep_ids:
                    retire(skill, "capacity overflow")
        if logs:
            state.maintenance_log.extend(logs)

    @staticmethod
    def _semantic_skill_similarity(
        left: ProceduralSkill,
        right: ProceduralSkill,
    ) -> float:
        left_text = " ".join(
            [
                left.activation_condition,
                *(step.action for step in left.execution_steps),
                left.termination_condition,
            ]
        )
        right_text = " ".join(
            [
                right.activation_condition,
                *(step.action for step in right.execution_steps),
                right.termination_condition,
            ]
        )
        return _jaccard(left_text, right_text)
