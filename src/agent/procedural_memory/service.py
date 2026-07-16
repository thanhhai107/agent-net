"""Skill-Pro Procedural Memory service.

This adapts the official Skill-Pro semantics to NIKA's diagnosis-agent
boundary: a Skill-MDP style selector injects active procedural skills before
diagnosis, while closed benchmark episodes feed a persistent trajectory buffer
and a provider-compatible evolution gate.
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
from agent.module_config import module_defaults
from agent.extensions.llm import load_extension_model as load_model
from agent.procedural_memory.safety import redact_oracle_markers
from agent.procedural_memory.attributes import infer_procedural_memory_attributes
from agent.procedural_memory.models import (
    EvaluationEvidence,
    ProceduralMemoryQuery,
    ProceduralMemoryState,
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

_DEFAULTS = module_defaults().procedural_memory
DEFAULT_POOL_SIZE = _DEFAULTS.pool_size
EXPERIENCE_POOL_SIZE = _DEFAULTS.experience_pool_size
PPO_EPSILON = _DEFAULTS.ppo_epsilon

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
BASELINE_EMA_ALPHA = _DEFAULTS.baseline_ema_alpha
SELECTION_EPSILON_DECAY_EPISODES = _DEFAULTS.selection_epsilon_decay_cases
SUPPORTED_VERIFIERS = frozenset(
    {"behavioral_replay", "policy_logprob", "structured_replay"}
)


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
        # Detection alone is not a reusable diagnosis procedure. Localization
        # and RCA therefore dominate the online learning signal.
        quality = (0.10 * detection) + (0.35 * localization) + (0.55 * rca)
    return _clamp(quality, 0.0, 1.0)


def _baseline_key(evidence: EvaluationEvidence) -> str:
    anomaly_class = (
        "anomaly"
        if evidence.ground_truth_is_anomaly is True
        else "clean"
        if evidence.ground_truth_is_anomaly is False
        else "unknown"
    )
    return "::".join(
        (
            evidence.scenario or "default",
            evidence.topology_class or "any-topology",
            anomaly_class,
        )
    )


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _skill_base_id(skill_id: str) -> str:
    return re.sub(r"_v\d+(?:_[a-f0-9]{6})?$", "", skill_id)


def _is_seed_skill(skill: ProceduralSkill) -> bool:
    return skill.skill_id in GENERIC_SEED_SKILL_IDS


def _learned_skill_unstable(skill: ProceduralSkill) -> bool:
    if skill.status == "retired":
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


def _sanitize_public_value(value: Any, evidence: EvaluationEvidence) -> Any:
    """Remove oracle/case-specific labels before replay persistence or prompting."""

    if isinstance(value, str):
        return _redact_hidden_labels(redact_oracle_markers(value), evidence)
    if isinstance(value, list):
        return [_sanitize_public_value(item, evidence) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_public_value(item, evidence)
            for key, item in value.items()
        }
    return value


def _sanitize_skill_step(step: SkillStep, evidence: EvaluationEvidence) -> SkillStep:
    return step.model_copy(
        deep=True,
        update={
            "action": _sanitize_public_value(step.action, evidence),
            "arguments_hint": _sanitize_public_value(step.arguments_hint, evidence),
            "observation_summary": _sanitize_public_value(
                step.observation_summary, evidence
            ),
            "rationale": _sanitize_public_value(step.rationale, evidence),
            "policy_state": _sanitize_public_value(step.policy_state, evidence),
            "policy_context": _sanitize_public_value(step.policy_context, evidence),
        },
    )


class ProceduralMemoryModule:
    def __init__(
        self,
        *,
        bank_id: str = "default",
        llm_backend: str | None = None,
        model: str | None = None,
        store_path: Path | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        evolution_threshold: int = _DEFAULTS.evolution_threshold,
        best_of_n: int = _DEFAULTS.best_of_n,
        ppo_epsilon: float = PPO_EPSILON,
        experience_pool_size: int = EXPERIENCE_POOL_SIZE,
        baseline_ema_alpha: float = BASELINE_EMA_ALPHA,
        selection_epsilon_decay_cases: int = SELECTION_EPSILON_DECAY_EPISODES,
        acceptance_margin: float = _DEFAULTS.acceptance_margin,
        verifier: str = _DEFAULTS.verifier,
        holdout_size: int = _DEFAULTS.holdout_size,
        min_positive_advantage: int = _DEFAULTS.min_positive_advantage,
        evolver_model: str = "",
        policy_scorer_model: str = "",
        policy_scorer: PolicyScorer | None = None,
        read_only: bool = False,
    ) -> None:
        self.bank_id = bank_id
        self.llm_backend = llm_backend
        self.model = model
        self.pool_size = pool_size
        self.evolution_threshold = evolution_threshold
        self.best_of_n = max(1, best_of_n)
        self.ppo_epsilon = ppo_epsilon
        self.experience_pool_size = max(1, experience_pool_size)
        self.baseline_ema_alpha = baseline_ema_alpha
        self.selection_epsilon_decay_cases = max(1, selection_epsilon_decay_cases)
        self.acceptance_margin = max(0.0, acceptance_margin)
        if verifier not in SUPPORTED_VERIFIERS:
            supported = ", ".join(sorted(SUPPORTED_VERIFIERS))
            raise ValueError(
                f"Unsupported verifier {verifier!r}; expected one of: {supported}"
            )
        self.verifier = verifier
        self.holdout_size = max(1, holdout_size)
        self.min_positive_advantage = max(0, min_positive_advantage)
        self.evolver_model = evolver_model.strip()
        self.policy_scorer_model = policy_scorer_model.strip()
        self._learning_llm_instance: Any | None = None
        self._policy_llm_instance: Any | None = None
        self.last_exploration_probability = 0.0
        self.last_exploration_arm = ""
        self.policy_scorer = policy_scorer or self._default_policy_scorer()
        self.store = ProceduralMemoryStore(
            bank_id=bank_id,
            state_path=store_path,
            read_only=read_only,
        )
        if not read_only:
            self._ensure_seed_skills()

    def _default_policy_scorer(self) -> PolicyScorer:
        selected_model = self._selected_policy_scorer_model()
        if self.verifier == "structured_replay":
            return StructuredReplayPolicyScorer()
        if self.verifier == "policy_logprob":
            if not selected_model:
                raise ValueError(
                    "policy_logprob verifier requires a policy scorer model"
                )
            base_url = os.getenv("NIKA_SKILL_LOGPROB_URL", "").strip()
            api_key = (
                os.getenv("NIKA_SKILL_LOGPROB_API_KEY", "").strip()
                or os.getenv("CUSTOM_API_KEY", "").strip()
            )
            if not base_url:
                raise ValueError(
                    "policy_logprob verifier requires NIKA_SKILL_LOGPROB_URL"
                )
            if not api_key:
                raise ValueError(
                    "policy_logprob verifier requires NIKA_SKILL_LOGPROB_API_KEY "
                    "or CUSTOM_API_KEY"
                )
            return PolicyLogprobScorer(
                base_url=base_url,
                api_key=api_key,
                model=selected_model,
                timeout=learning_timeout_seconds(),
            )
        return BehavioralReplayPolicyScorer(self._policy_llm)

    def _learning_llm(self) -> Any | None:
        selected_backend = self._selected_learning_backend()
        selected_model = self._selected_evolver_model()
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

    def _policy_llm(self) -> Any | None:
        selected_backend = self._selected_learning_backend()
        selected_model = self._selected_policy_scorer_model()
        if not selected_backend or not selected_model:
            return None
        if self._policy_llm_instance is None:
            self._policy_llm_instance = load_model(
                selected_backend,
                selected_model,
                timeout=learning_timeout_seconds(),
                max_retries=learning_max_retries(),
            )
        return self._policy_llm_instance

    def _selected_evolver_model(self) -> str:
        return self.evolver_model or learning_model(self.model) or self.model or ""

    def _selected_learning_backend(self) -> str:
        if not self.llm_backend:
            return ""
        return learning_backend(self.llm_backend)

    def _selected_policy_scorer_model(self) -> str:
        return (
            self.policy_scorer_model
            or os.getenv("NIKA_SKILL_LOGPROB_MODEL", "").strip()
            or module_defaults().procedural_memory.skill_logprob_model
            or learning_model(self.model)
            or self.model
            or ""
        )

    def clear(self) -> None:
        self.store.clear()
        self._ensure_seed_skills()

    def _save_state(self, state: ProceduralMemoryState) -> None:
        """Persist a bounded bank while preserving the active Skill pool."""

        history_limit = max(self.experience_pool_size, self.pool_size * 4, 100)
        state.episodes = state.episodes[-history_limit:]
        state.experiences = self._bounded_experiences(state.experiences)
        state.ppo_decisions = state.ppo_decisions[-history_limit:]
        state.evolution_log = state.evolution_log[-history_limit:]
        state.maintenance_log = state.maintenance_log[-history_limit:]

        for skill in state.skills.values():
            skill.source_sessions = skill.source_sessions[-history_limit:]
            skill.semantic_gradients = skill.semantic_gradients[-history_limit:]

        retired_learned = sorted(
            (
                skill
                for skill in state.skills.values()
                if skill.status == "retired" and not _is_seed_skill(skill)
            ),
            key=lambda skill: (skill.updated_at, skill.created_at, skill.skill_id),
            reverse=True,
        )
        retired_limit = max(self.pool_size, 16)
        for skill in retired_learned[retired_limit:]:
            state.skills.pop(skill.skill_id, None)

        self.store.save(state)

    def _bounded_experiences(
        self,
        experiences: list[SkillExperience],
    ) -> list[SkillExperience]:
        if len(experiences) <= self.experience_pool_size:
            return experiences
        indexed = list(enumerate(experiences))
        unconsumed = [item for item in indexed if not item[1].used_for_evolution]
        consumed = [item for item in indexed if item[1].used_for_evolution]
        kept = unconsumed[-self.experience_pool_size :]
        remaining = self.experience_pool_size - len(kept)
        if remaining > 0:
            kept.extend(consumed[-remaining:])
        kept.sort(key=lambda item: item[0])
        return [experience for _, experience in kept]

    def _ensure_seed_skills(self) -> None:
        with self.store.exclusive():
            state = self.store.load()
            changed = self.store.needs_migration
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
                    if stored.status not in {"validated", "retired"}:
                        stored.status = "validated"
                        changed = True
            if changed:
                self._save_state(state)

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
        self,
        *,
        query: ProceduralMemoryQuery,
        include_probationary: bool = False,
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
            allowed_statuses = (
                {"validated", "probationary"} if include_probationary else {"validated"}
            )
            if skill.status not in allowed_statuses:
                continue
            if _learned_skill_unstable(skill):
                continue
            reasons: list[str] = []
            score = self._skill_effective_score(skill)
            if skill.status == "probationary":
                score += 0.20 / math.sqrt(skill.frequency + 1.0)
                reasons.append("probation_uncertainty")
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

    def selection_candidates(
        self,
        *,
        include_probationary: bool = False,
        exclude_skill_ids: Collection[str] | None = None,
    ) -> list[ProceduralSkill]:
        """Return the complete eligible pool for direct LLM selection."""

        excluded = {skill_id for skill_id in (exclude_skill_ids or []) if skill_id}
        allowed = (
            {"validated", "probationary"} if include_probationary else {"validated"}
        )
        return sorted(
            (
                skill
                for skill in self.store.load().skills.values()
                if skill.status in allowed
                and skill.skill_id not in excluded
                and not _learned_skill_unstable(skill)
            ),
            key=lambda skill: skill.skill_id,
        )

    def activate_skill(
        self,
        skill_id: str,
        *,
        record_reuse: bool = True,
        include_probationary: bool = False,
        exclude_skill_ids: Collection[str] | None = None,
    ) -> SkillRetrieval | None:
        """Validate and activate exactly one selector-provided skill id."""

        candidates = {
            skill.skill_id: skill
            for skill in self.selection_candidates(
                include_probationary=include_probationary,
                exclude_skill_ids=exclude_skill_ids,
            )
        }
        skill = candidates.get(skill_id)
        if skill is None:
            return None
        selected = SkillRetrieval(
            skill=skill,
            score=self._skill_effective_score(skill),
            reasons=["llm_direct_selection"],
        )
        return self._record_skill_reuse(selected) if record_reuse else selected

    def exploration_selection(
        self,
        *,
        epsilon: float,
        key: str,
        query: ProceduralMemoryQuery | None = None,
        record_reuse: bool = True,
        exclude_skill_ids: Collection[str] | None = None,
    ) -> tuple[bool, SkillRetrieval | None]:
        """Apply reproducible, context-filtered exploration with probation priority."""

        if not self._exploration_triggered(epsilon, key):
            self.last_exploration_probability = max(0.0, 1.0 - epsilon)
            self.last_exploration_arm = "llm_controller"
            return False, None
        excluded = {skill_id for skill_id in (exclude_skill_ids or []) if skill_id}
        if query is None:
            candidates = [
                SkillRetrieval(
                    skill=skill,
                    score=self._skill_effective_score(skill),
                    reasons=[],
                )
                for skill in self.selection_candidates(
                    include_probationary=True,
                    exclude_skill_ids=excluded,
                )
            ]
        else:
            candidates = [
                item
                for item in self.retrieve(query=query, include_probationary=True)
                if item.skill.skill_id not in excluded
            ]
        weighted: list[tuple[SkillRetrieval | None, float]] = []
        for item in candidates:
            if item.skill.status == "probationary":
                weight = 4.0 / math.sqrt(item.skill.frequency + 1.0)
            else:
                weight = 1.0 / math.sqrt(item.skill.frequency + 1.0)
            weighted.append((item, weight))
        # Keep an explicit no-skill arm without letting it dominate small pools.
        weighted.append((None, 0.5))
        total_weight = sum(weight for _, weight in weighted)
        draw = self._stable_unit_interval(f"{key}:choice") * total_weight
        choice: SkillRetrieval | None = None
        choice_weight = 0.5
        cumulative = 0.0
        for item, weight in weighted:
            cumulative += weight
            if draw <= cumulative:
                choice = item
                choice_weight = weight
                break
        self.last_exploration_probability = epsilon * choice_weight / total_weight
        self.last_exploration_arm = (
            choice.skill.skill_id if choice is not None else "no_skill"
        )
        if choice is None:
            return True, None
        selected = SkillRetrieval(
            skill=choice.skill,
            score=choice.score,
            reasons=[
                *choice.reasons,
                "contextual_probation_exploration"
                if choice.skill.status == "probationary"
                else "contextual_epsilon_exploration",
            ],
        )
        return True, self._record_skill_reuse(selected) if record_reuse else selected

    @staticmethod
    def _stable_unit_interval(value: str) -> float:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _exploration_triggered(self, epsilon: float, key: str) -> bool:
        return bool(
            epsilon > 0 and key and self._stable_unit_interval(f"{key}:gate") < epsilon
        )

    def decayed_selection_epsilon(self, initial: float) -> float:
        iteration = self.store.load().iteration
        progress = min(
            max(iteration, 0) / float(self.selection_epsilon_decay_cases),
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
                self._save_state(state)
                selected.skill = stored
        return selected

    def snapshot(self, *, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(self.store.snapshot_jsonl()) + "\n", encoding="utf-8"
        )
        return output_path

    def bank_state_hash(self) -> str:
        if not self.store.state_path.exists():
            return ""
        return hashlib.sha256(self.store.state_path.read_bytes()).hexdigest()

    def freeze_for_evaluation(self, *, output_path: Path) -> dict[str, Any]:
        """Retire unresolved probationary skills and snapshot a read-only bank."""

        retired_ids: list[str] = []
        with self.store.exclusive():
            state = self.store.load()
            for skill in state.skills.values():
                if skill.status != "probationary":
                    continue
                skill.status = "retired"
                retired_ids.append(skill.skill_id)
                state.maintenance_log.append(
                    {
                        "stage": "freeze unresolved probationary skill",
                        "skill_id": skill.skill_id,
                        "frequency": skill.frequency,
                        "avg_gain": skill.avg_gain,
                    }
                )
            if retired_ids:
                self._save_state(state)
        snapshot_path = self.snapshot(output_path=output_path)
        state = self.store.load()
        return {
            "bank_id": self.bank_id,
            "iteration": state.iteration,
            "state_hash": self.bank_state_hash(),
            "snapshot_path": str(snapshot_path),
            "retired_probationary_skill_ids": sorted(retired_ids),
            "validated_skill_ids": sorted(
                skill.skill_id
                for skill in state.skills.values()
                if skill.status == "validated"
            ),
        }

    def propose_skill(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        parent: ProceduralSkill | None = None,
        candidate_index: int = 0,
        critique: SemanticGradient | None = None,
        experiences: list[SkillExperience] | None = None,
        sampled_candidate: SkillCandidateDraft | None = None,
    ) -> ProceduralSkill:
        if not tool_steps:
            raise ValueError("Skill-Pro requires at least one observed execution step.")
        critique = (
            critique.model_copy(deep=True)
            if critique is not None
            else self.semantic_gradient(evidence=evidence, tool_steps=tool_steps)
        )
        sampled = sampled_candidate
        if sampled is None:
            sampled = self._llm_skill_candidate(
                evidence=evidence,
                parent=parent,
                critique=critique,
                experiences=experiences or [],
                candidate_index=candidate_index,
            )
        if sampled is None:
            raise RuntimeError(
                "Procedural Memory candidate evolver returned no valid skill."
            )
        batch_text = " ".join(
            [
                _episode_attribute_text(evidence, tool_steps),
                sampled.title,
                sampled.initiation,
                " ".join(sampled.policy),
                sampled.termination,
                *(item.trajectory for item in (experiences or []) if item.trajectory),
            ]
        )
        attrs = infer_procedural_memory_attributes(
            batch_text,
            tools=[step.tool_name for step in tool_steps if step.tool_name],
        )
        source_session_ids = sorted(
            {item.session_id for item in (experiences or []) if item.session_id}
        ) or [evidence.session_id]
        candidate_content = {
            "title": " ".join(sampled.title.lower().split()),
            "initiation": " ".join(sampled.initiation.lower().split()),
            "policy": [" ".join(step.lower().split()) for step in sampled.policy],
            "termination": " ".join(sampled.termination.lower().split()),
        }
        if parent is None:
            skill_id = _stable_id(
                attrs.protocols,
                attrs.services,
                attrs.symptoms,
                attrs.tools,
                [step.tool_name for step in tool_steps if step.tool_name],
                candidate_content,
                prefix="skill",
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
                        source_session_ids,
                        attrs.protocols,
                        attrs.services,
                        attrs.symptoms,
                        attrs.tools,
                        candidate_content,
                        candidate_index,
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()[:6]
            skill_id = f"{base}_v{version}_{revision}"
            parent_id = parent.skill_id
        title = sampled.title.strip()
        activation = sampled.initiation.strip()
        steps = [
            SkillStep(
                order=index + 1,
                action=action.strip(),
                rationale="Independently sampled Skill-Pro candidate.",
            )
            for index, action in enumerate(sampled.policy[:8])
            if action.strip()
        ]
        termination = sampled.termination.strip()
        outcome_success = _metric_success(
            evidence.metrics,
            evidence.ground_truth_is_anomaly,
        )
        return ProceduralSkill(
            skill_id=skill_id,
            title=title,
            activation_condition=activation,
            execution_steps=steps[:10],
            termination_condition=termination,
            source_sessions=source_session_ids,
            scenarios=[],
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

    def _skill_candidate_prompt(
        self,
        *,
        parent: ProceduralSkill | None,
        critique: SemanticGradient,
        experiences: list[SkillExperience],
        output_instruction: str,
    ) -> str:
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
        return (
            "You are the Skill-Pro Skill Evolver. Produce procedural candidates "
            "from a batch-aggregated semantic "
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
            f"Parent skill:\n{parent_payload}\n\n"
            "Aggregated semantic gradient:\n"
            f"{critique.model_dump_json(indent=2)}\n\n"
            "Batch evidence summaries:\n"
            f"{json.dumps(batch_payload, indent=2, ensure_ascii=False)}\n\n"
            f"{output_instruction}"
        )

    @staticmethod
    def _redact_semantic_gradient(
        gradient: SemanticGradient,
        evidence: EvaluationEvidence,
    ) -> SemanticGradient:
        redacted = gradient.model_copy(deep=True)
        redacted.critique = _redact_hidden_labels(
            redact_oracle_markers(redacted.critique), evidence
        )
        redacted.proposed_update = _redact_hidden_labels(
            redact_oracle_markers(redacted.proposed_update), evidence
        )
        redacted.component_update.initiation = _redact_hidden_labels(
            redact_oracle_markers(redacted.component_update.initiation), evidence
        )
        redacted.component_update.policy = [
            _redact_hidden_labels(redact_oracle_markers(step), evidence)
            for step in redacted.component_update.policy
        ]
        redacted.component_update.termination = _redact_hidden_labels(
            redact_oracle_markers(redacted.component_update.termination), evidence
        )
        return redacted

    @staticmethod
    def _redact_skill_candidate(
        draft: SkillCandidateDraft,
        evidence: EvaluationEvidence,
    ) -> SkillCandidateDraft | None:
        redacted = draft.model_copy(deep=True)
        redacted.title = _redact_hidden_labels(
            redact_oracle_markers(redacted.title), evidence
        ).strip()
        redacted.initiation = _redact_hidden_labels(
            redact_oracle_markers(redacted.initiation), evidence
        ).strip()
        redacted.policy = [
            value
            for step in redacted.policy
            if (
                value := _redact_hidden_labels(
                    redact_oracle_markers(step), evidence
                ).strip()
            )
        ]
        redacted.termination = _redact_hidden_labels(
            redact_oracle_markers(redacted.termination), evidence
        ).strip()
        if not all(
            (
                redacted.title,
                redacted.initiation,
                redacted.policy,
                redacted.termination,
            )
        ):
            return None
        return redacted

    def _llm_skill_candidate(
        self,
        *,
        evidence: EvaluationEvidence,
        parent: ProceduralSkill | None,
        critique: SemanticGradient,
        experiences: list[SkillExperience],
        candidate_index: int,
    ) -> SkillCandidateDraft | None:
        """Sample one candidate for direct single-candidate callers."""
        if not self._selected_learning_backend() or not self._selected_evolver_model():
            return None
        prompt = self._skill_candidate_prompt(
            parent=parent,
            critique=critique,
            experiences=experiences,
            output_instruction=(
                f"Candidate sample index: {candidate_index}. Return "
                "SkillCandidateDraft with a concise title, observable initiation "
                "condition, 3-6 executable policy steps, and a checkable termination "
                "condition."
            ),
        )
        try:
            llm = self._learning_llm()
            if llm is None:
                raise RuntimeError("learning LLM is unavailable")
            evolver = llm.with_structured_output(SkillCandidateDraft)
            raw = evolver.invoke(prompt)
            draft = (
                raw
                if isinstance(raw, SkillCandidateDraft)
                else SkillCandidateDraft.model_validate(raw)
            )
            candidate = self._redact_skill_candidate(draft, evidence)
            if candidate is None:
                raise RuntimeError("candidate evolver returned an incomplete skill")
            return candidate
        except Exception as exc:
            raise RuntimeError(format_learning_error(exc)) from exc

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
        raise RuntimeError(
            llm_error or "Procedural Memory semantic-gradient LLM is not configured."
        )

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
            topology_class=experience.topology_class,
            metrics={
                **experience.metrics,
                "episode_reward": experience.reward,
                "episode_baseline": experience.baseline,
                "episode_advantage": experience.advantage,
                "skill_credit_weight": experience.credit_weight,
            },
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
        experiences: list[SkillExperience] | None = None,
    ) -> SemanticGradient:
        """Consolidate per-trajectory gradients into one stable batch update."""
        if not gradients:
            raise RuntimeError("No trajectory gradients were available for evolution.")
        if not self._selected_learning_backend() or not self._selected_evolver_model():
            raise RuntimeError(
                "Procedural Memory semantic-gradient LLM is not configured."
            )
        experience_by_session = {item.session_id: item for item in (experiences or [])}
        gradient_payload = []
        for gradient in gradients:
            sample = experience_by_session.get(gradient.source_session_id)
            gradient_payload.append(
                {
                    "gradient": gradient.model_dump(mode="json"),
                    "reward": sample.reward if sample is not None else None,
                    "advantage": sample.advantage if sample is not None else None,
                    "credit_weight": (
                        sample.credit_weight if sample is not None else None
                    ),
                }
            )
        prompt = (
            "You are the Skill-Pro batch semantic-gradient aggregator. "
            "Consolidate recurring, causally supported updates and discard "
            "conflicting or trajectory-specific details. Do not include device "
            "names, hidden labels, benchmark ids, or exact one-off values.\n\n"
            "Parent skill:\n"
            f"{skill.format_for_llm() if skill else 'NEW SKILL'}\n\n"
            "Per-trajectory gradients:\n"
            f"{json.dumps(gradient_payload, indent=2, ensure_ascii=False)}\n\n"
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
                gradient.critique = _redact_hidden_labels(gradient.critique, evidence)
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
            raise RuntimeError(format_learning_error(exc)) from exc

    def _llm_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
        skill: ProceduralSkill | None = None,
    ) -> tuple[SemanticGradient | None, str]:
        selected_backend = self._selected_learning_backend()
        selected_model = self._selected_evolver_model()
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
            "episode_reward",
            "episode_baseline",
            "episode_advantage",
            "skill_credit_weight",
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
        sample_batch = samples or [
            SkillExperience(
                experience_id=_stable_id(evidence.session_id, "gate", prefix="exp"),
                session_id=evidence.session_id,
                reward=current_reward,
                baseline=baseline_score,
                advantage=current_reward - baseline_score,
                success=_metric_success(
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
        margin = self.acceptance_margin
        parent_safe = baseline is None or not _learned_skill_unstable(baseline)
        verified_success_count = sum(item.success for item in sample_batch)
        positive_advantage_count = sum(item.advantage > 0 for item in sample_batch)
        verification_error = str(replay.get("verification_error") or "")
        has_positive_support = positive_advantage_count >= self.min_positive_advantage
        replay_no_alignment_regression = (
            replay["candidate_alignment"] + margin >= replay["baseline_alignment"]
        )
        verification_method = str(
            replay.get("verification_method") or "structured_replay"
        )
        structured_check = StructuredReplayPolicyScorer().score_batch(
            candidate=candidate,
            baseline=baseline,
            experiences=sample_batch,
        )
        structured_candidate = sum(
            item.candidate_alignment for item in structured_check.scores
        ) / max(len(structured_check.scores), 1)
        structured_baseline = sum(
            item.baseline_alignment for item in structured_check.scores
        ) / max(len(structured_check.scores), 1)
        structured_no_regression = structured_candidate + margin >= structured_baseline
        no_alignment_regression = structured_no_regression and (
            replay_no_alignment_regression
            if verification_method != "policy_logprob"
            else True
        )
        accepted = (
            not verification_error
            and j_score > margin
            and parent_safe
            and has_positive_support
            and no_alignment_regression
        )
        gate_name = (
            "policy log-prob replay gate"
            if verification_method == "policy_logprob"
            else verification_method.replace("_", " ") + " gate"
        )
        reason = (
            f"candidate passed {gate_name}"
            if accepted
            else "candidate verification deferred: verifier unavailable"
            if verification_error
            else f"candidate failed {gate_name}: unstable parent skill"
            if not parent_safe
            else f"candidate failed {gate_name}: insufficient positive support"
            if not has_positive_support
            else f"candidate failed {gate_name}: alignment regression"
            if not no_alignment_regression
            else f"candidate failed {gate_name}"
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
            parent_j_score=float(replay.get("parent_j_score") or 0.0),
            delta_j_score=j_score,
            candidate_alignment=replay["candidate_alignment"],
            baseline_alignment=replay["baseline_alignment"],
            sample_count=len(sample_batch),
            best_of_n=best_of_n,
            candidate_type="REFINE" if candidate_type == "REFINE" else "NEW",
            verification_method=verification_method,
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
        total_added_tokens = int(
            evidence.metrics.get("procedural_memory_total_added_tokens") or 0
        )
        delta_prompt_tokens_per_step = total_added_tokens / max(
            evidence.steps or len(tool_steps), 1
        )
        reward = _evidence_score(evidence)
        baseline_key = _baseline_key(evidence)
        baseline_value = state.baselines.get(
            baseline_key,
            state.baselines.get(evidence.scenario or "default", 0.0),
        )
        promotion_safe = _metric_success(
            evidence.metrics,
            evidence.ground_truth_is_anomaly,
        )
        runtime_skill_counts = self._runtime_skill_counts(state, tool_steps)
        experience_skill_ids = sorted(runtime_skill_counts)
        attributed_steps = sum(
            bool(
                step.skill_id
                and step.activation_id
                and step.skill_id in runtime_skill_counts
            )
            for step in tool_steps
        )
        unattributed_steps = max(0, len(tool_steps) - attributed_steps)
        attribution_rate = attributed_steps / max(len(tool_steps), 1)
        segment_experiences = self._segment_experiences(
            evidence=evidence,
            tool_steps=tool_steps,
            reward=reward,
            baseline=baseline_value,
            success=promotion_safe,
            valid_skill_ids=set(runtime_skill_counts),
        )
        episode_is_new = not any(
            item.session_id == evidence.session_id for item in state.episodes
        )
        if episode_is_new:
            state.episodes.append(public_episode_evidence(evidence))
        existing_experience_ids = {item.experience_id for item in state.experiences}
        current_experience_ids = sorted(
            segment.experience_id for segment in segment_experiences.values()
        )
        if episode_is_new:
            for segment in segment_experiences.values():
                if segment.experience_id in existing_experience_ids:
                    continue
                state.experiences.append(segment)
                existing_experience_ids.add(segment.experience_id)
        state.experiences = self._bounded_experiences(state.experiences)
        if episode_is_new:
            self._update_baseline(state, baseline_key, reward)
            state.iteration += 1
            for skill in state.skills.values():
                if skill.status != "retired":
                    skill.increment_maturity()
            total_calls = max(sum(runtime_skill_counts.values()), 1)
            for skill_id, count in runtime_skill_counts.items():
                state.skills[skill_id].update_stats(
                    reward=reward,
                    baseline=baseline_value,
                    total_skill_calls=total_calls,
                    skill_call_count=count,
                )

        current_experience = next(iter(segment_experiences.values()), None)

        def report_base() -> dict[str, Any]:
            return {
                "episode_reward": reward,
                "episode_baseline": baseline_value,
                "episode_advantage": reward - baseline_value,
                "episode_success": promotion_safe,
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
                "skills": len(state.skills),
                "experience_id": (
                    current_experience.experience_id if current_experience else ""
                ),
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "experience_ids": current_experience_ids,
                "attributed_steps": attributed_steps,
                "unattributed_steps": unattributed_steps,
                "attribution_rate": round(attribution_rate, 6),
                "method": "Skill-Pro",
            }

        # Persist episode accounting before potentially slow LLM evolution calls.
        self._save_state(state)
        required_batch_size = max(self.evolution_threshold, 2)
        evolution_parent = self._next_evolution_parent(state)
        samples = self._evolution_batch(state, evolution_parent)
        if len(samples) < required_batch_size:
            reason = (
                "no Skill-Pro trajectory batch is ready"
                if evolution_parent is None
                else "insufficient Skill-Pro evolution batch"
            )
            self._maintain(state)
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": evolution_parent.skill_id if evolution_parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "candidate": "",
                    "action": "deferred",
                    "reason": reason,
                    "sample_count": len(samples),
                    "required_sample_count": required_batch_size,
                    "attributed_steps": attributed_steps,
                    "unattributed_steps": unattributed_steps,
                    "attribution_rate": round(attribution_rate, 6),
                }
            )
            self._save_state(state)
            return {
                "status": "deferred",
                "reason": reason,
                "skill_id": evolution_parent.skill_id if evolution_parent else "",
                "semantic_gradient_source": "pending",
                "semantic_gradient_llm_attempted": False,
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_llm_error": "",
                "semantic_gradient_count": 0,
                "decision": None,
                "sample_count": len(samples),
                "required_sample_count": required_batch_size,
                **report_base(),
            }

        verification_samples = self._verification_batch(
            state,
            generation_samples=samples,
        )
        verification_session_ids = {
            sample.session_id for sample in verification_samples
        }
        if len(samples) > 1:
            samples = [
                sample
                for sample in samples
                if sample.session_id not in verification_session_ids
            ]
        if not verification_samples:
            reason = "insufficient disjoint Skill-Pro verification batch"
            self._maintain(state)
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": evolution_parent.skill_id if evolution_parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "sample_experience_ids": [
                        sample.experience_id for sample in samples
                    ],
                    "verification_experience_ids": [],
                    "candidate": "",
                    "action": "deferred",
                    "reason": reason,
                    "sample_count": len(samples),
                    "required_sample_count": required_batch_size,
                    "required_verification_count": 1,
                    "attributed_steps": attributed_steps,
                    "unattributed_steps": unattributed_steps,
                    "attribution_rate": round(attribution_rate, 6),
                }
            )
            self._save_state(state)
            return {
                "status": "deferred",
                "reason": reason,
                "skill_id": evolution_parent.skill_id,
                "semantic_gradient_source": "pending",
                "semantic_gradient_llm_attempted": False,
                "semantic_gradient_llm_failed": False,
                "semantic_gradient_llm_error": "",
                "semantic_gradient_count": 0,
                "verification_method": "none",
                "verification_error": "",
                "decision": None,
                "sample_count": len(samples),
                "required_sample_count": required_batch_size,
                "verification_sample_count": 0,
                "required_verification_count": 1,
                **report_base(),
            }
        target_tool_steps = self._tool_steps_from_experiences(samples)
        if not target_tool_steps:
            raise RuntimeError("Skill-Pro evolution batch contains no actions")
        candidate_attempts: list[
            tuple[
                ProceduralSkill,
                PPOGateDecision,
                list[SemanticGradient],
                int,
                float,
                str,
            ]
        ] = []
        candidate_errors: list[str] = []
        gradients: list[SemanticGradient] = []
        related_count = 0
        relevance = 0.0
        candidate_type = "REFINE"
        candidate_parent: ProceduralSkill | None = evolution_parent
        batch_gradient: SemanticGradient | None = None
        try:
            for sample in samples:
                gradient = self.semantic_gradient_from_experience(
                    experience=sample,
                    skill=evolution_parent,
                )
                gradients.append(self._redact_semantic_gradient(gradient, evidence))
            related_count = sum(
                gradient.component_update.is_related for gradient in gradients
            )
            relevance = related_count / max(len(gradients), 1)
            candidate_type = (
                "REFINE"
                if evolution_parent is not None
                and not _learned_skill_unstable(evolution_parent)
                and relevance >= 0.5
                else "NEW"
            )
            candidate_parent = evolution_parent if candidate_type == "REFINE" else None
            batch_gradient = self.aggregate_semantic_gradients(
                gradients=gradients,
                skill=candidate_parent,
                source_session_id=_stable_id(
                    *(sample.session_id for sample in samples),
                    prefix="batch",
                ),
                evidence=evidence,
                experiences=samples,
            )
        except Exception as exc:
            candidate_errors.append(f"semantic gradient: {format_learning_error(exc)}")

        for index in range(self.best_of_n if batch_gradient is not None else 0):
            try:
                sampled_candidate = self._llm_skill_candidate(
                    evidence=evidence,
                    parent=candidate_parent,
                    critique=batch_gradient,
                    experiences=samples,
                    candidate_index=index,
                )
                if sampled_candidate is None:
                    raise RuntimeError(
                        "Procedural Memory candidate evolver returned no valid skill."
                    )
                candidate = self.propose_skill(
                    evidence=evidence,
                    tool_steps=target_tool_steps,
                    parent=candidate_parent,
                    candidate_index=index,
                    critique=batch_gradient,
                    experiences=samples,
                    sampled_candidate=sampled_candidate,
                )
                if candidate_type == "NEW" and evolution_parent is not None:
                    candidate.parent_id = evolution_parent.skill_id
                candidate.status = "candidate"
                candidate.success_count = 0
                candidate.failure_count = 0
                candidate.score = (
                    self._skill_effective_score(candidate_parent)
                    if candidate_parent is not None
                    else 0.0
                )
                candidate.prior_score = candidate.score
                decision = self.ppo_gate(
                    candidate=candidate,
                    evidence=evidence,
                    baseline=evolution_parent,
                    samples=verification_samples,
                    candidate_type=candidate_type,
                    best_of_n=self.best_of_n,
                )
                candidate_attempts.append(
                    (
                        candidate,
                        decision,
                        gradients,
                        related_count,
                        relevance,
                        candidate_type,
                    )
                )
            except Exception as exc:
                candidate_errors.append(
                    f"candidate {index}: {format_learning_error(exc)}"
                )

        valid_attempts = [
            attempt
            for attempt in candidate_attempts
            if not attempt[1].verification_error
        ]
        if not valid_attempts:
            verification_errors = [
                attempt[1].verification_error
                for attempt in candidate_attempts
                if attempt[1].verification_error
            ]
            errors = [*candidate_errors, *verification_errors]
            reason = "Procedural Memory evolution deferred: " + (
                " | ".join(dict.fromkeys(errors))
                if errors
                else "no candidate completed verification"
            )
            state.evolution_log.append(
                {
                    "iteration": state.iteration,
                    "parent": evolution_parent.skill_id if evolution_parent else "",
                    "runtime_skill_ids": sorted(set(experience_skill_ids)),
                    "generation_experience_ids": [
                        sample.experience_id for sample in samples
                    ],
                    "verification_experience_ids": [
                        sample.experience_id for sample in verification_samples
                    ],
                    "candidate": "",
                    "action": "deferred",
                    "reason": reason,
                    "best_of_n": self.best_of_n,
                    "candidate_attempt_count": len(candidate_attempts),
                    "candidate_errors": candidate_errors,
                    "attributed_steps": attributed_steps,
                    "unattributed_steps": unattributed_steps,
                    "attribution_rate": round(attribution_rate, 6),
                }
            )
            self._maintain(state)
            self._save_state(state)
            return {
                "status": "deferred",
                "reason": reason,
                "skill_id": evolution_parent.skill_id if evolution_parent else "",
                "semantic_gradient_source": "llm",
                "semantic_gradient_llm_attempted": True,
                "semantic_gradient_llm_failed": bool(candidate_errors),
                "semantic_gradient_llm_error": " | ".join(candidate_errors),
                "semantic_gradient_count": len(gradients),
                "verification_method": (
                    candidate_attempts[0][1].verification_method
                    if candidate_attempts
                    else "none"
                ),
                "verification_error": " | ".join(dict.fromkeys(verification_errors)),
                "verification_sample_count": len(verification_samples),
                "decision": None,
                **report_base(),
            }

        accepted_attempts = [
            attempt for attempt in valid_attempts if attempt[1].accepted
        ]
        selection_pool = accepted_attempts or valid_attempts
        (
            best_candidate,
            best_decision,
            per_trajectory_gradients,
            related_gradient_count,
            relevance_ratio,
            candidate_type,
        ) = max(selection_pool, key=lambda attempt: attempt[1].j_score)
        state.ppo_decisions.append(best_decision)
        gradient_source = "llm"
        gradient_error = ""
        if best_decision.accepted:
            # Offline replay is an admissibility gate. Publication requires
            # independent online reward while the candidate is probationary.
            best_candidate.status = "probationary"
            best_candidate.source_sessions = sorted(
                set(best_candidate.source_sessions)
                | {
                    sample.session_id
                    for sample in [*samples, *verification_samples]
                    if sample.session_id
                }
            )
            if evolution_parent is not None and candidate_type == "REFINE":
                best_candidate.source_sessions = sorted(
                    set(
                        evolution_parent.source_sessions
                        + best_candidate.source_sessions
                    )
                )
                best_candidate.semantic_gradients = (
                    evolution_parent.semantic_gradients
                    + best_candidate.semantic_gradients
                )
            old = state.skills.get(best_candidate.skill_id)
            if old is not None:
                best_candidate.reuse_count = old.reuse_count
                best_candidate.frequency = old.frequency
                best_candidate.episode_exposures = old.episode_exposures
                best_candidate.activation_count = old.activation_count
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
                evolution_parent is not None
                and candidate_type == "REFINE"
                and evolution_parent.skill_id in state.skills
            ):
                state.skills[
                    evolution_parent.skill_id
                ].last_evolved_iteration = state.iteration
            state.skills[best_candidate.skill_id] = best_candidate
        generation_sample_ids = [sample.experience_id for sample in samples]
        consumed_sample_ids = list(
            dict.fromkeys(
                [
                    *generation_sample_ids,
                    *(sample.experience_id for sample in verification_samples),
                ]
            )
        )
        consumed_sample_id_set = set(consumed_sample_ids)
        consumed_session_order = list(
            dict.fromkeys(
                sample.session_id for sample in [*samples, *verification_samples]
            )
        )
        consumed_session_ids = set(consumed_session_order)
        retained_session_ids: set[str] = set()
        if not best_decision.accepted:
            retain_count = max(1, len(consumed_session_ids) // 2)
            retained_session_ids = set(consumed_session_order[-retain_count:])
        for item in state.experiences:
            if (
                item.experience_id in consumed_sample_id_set
                or (
                    item.session_id in consumed_session_ids
                    and evolution_parent.skill_id in item.skill_ids
                )
            ) and item.session_id not in retained_session_ids:
                item.used_for_evolution = True
        state.evolution_log.append(
            {
                "iteration": state.iteration,
                "parent": evolution_parent.skill_id if evolution_parent else "",
                "runtime_skill_ids": sorted(set(experience_skill_ids)),
                "sample_experience_ids": consumed_sample_ids,
                "generation_experience_ids": generation_sample_ids,
                "generation_session_ids": [sample.session_id for sample in samples],
                "verification_experience_ids": [
                    sample.experience_id for sample in verification_samples
                ],
                "verification_session_ids": [
                    sample.session_id for sample in verification_samples
                ],
                "candidate": best_candidate.skill_id,
                "action": ("accepted" if best_decision.accepted else "rejected"),
                "publication_status": (
                    "probationary" if best_decision.accepted else "not_published"
                ),
                "j_score": best_decision.j_score,
                "candidate_alignment": best_decision.candidate_alignment,
                "baseline_alignment": best_decision.baseline_alignment,
                "sample_count": best_decision.sample_count,
                "best_of_n": self.best_of_n,
                "candidate_attempt_count": len(candidate_attempts),
                "candidate_errors": candidate_errors,
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
        self._save_state(state)
        return {
            "status": ("accepted" if best_decision.accepted else "rejected"),
            "reason": best_decision.reason,
            "skill_id": best_candidate.skill_id,
            "semantic_gradient_source": gradient_source,
            "semantic_gradient_llm_attempted": bool(
                self._selected_learning_backend() and self._selected_evolver_model()
            ),
            "semantic_gradient_llm_failed": bool(gradient_error),
            "semantic_gradient_llm_error": gradient_error,
            "semantic_gradient_count": len(per_trajectory_gradients),
            "related_semantic_gradient_count": related_gradient_count,
            "relevance_ratio": round(relevance_ratio, 6),
            "candidate_type": candidate_type,
            "verification_method": best_decision.verification_method,
            "verification_error": best_decision.verification_error,
            "verification_sample_count": len(verification_samples),
            "decision": best_decision.model_dump(),
            "skill_status": best_candidate.status,
            **report_base(),
        }

    def _runtime_skill_counts(
        self,
        state,
        tool_steps: list[SkillStep],
    ) -> Counter[str]:
        valid_steps = [
            step
            for step in tool_steps
            if step.skill_id and step.activation_id and step.skill_id in state.skills
        ]
        activations = {(step.skill_id, step.activation_id) for step in valid_steps}
        return Counter(skill_id for skill_id, _ in activations)

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
        staleness = max(
            1,
            state.iteration
            - (skill.last_evolved_iteration or max(skill.maturity - frequency, 0)),
        )
        staleness_factor = 1.0 - math.exp(-staleness / 8.0)
        uncertainty = 1.0 / math.sqrt(frequency + 1.0)
        return (
            impact * max(confidence, 0.1) * gap
            + 0.05 * staleness_factor
            + 0.02 * uncertainty
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
        transitions: list[SkillTransition] | None = None,
        credit_weight: float = 1.0,
    ) -> SkillExperience:
        transitions = transitions or self._transitions_from_steps(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        experience = SkillExperience(
            experience_id=_stable_id(
                evidence.session_id,
                skill_ids,
                [step.model_dump(mode="json") for step in tool_steps],
                prefix="exp",
            ),
            session_id=evidence.session_id,
            reward=reward,
            baseline=baseline,
            advantage=(reward - baseline) * credit_weight,
            skill_ids=skill_ids,
            trajectory=_sanitize_public_value(evidence.task_description, evidence),
            scenario=evidence.scenario,
            topology_class=evidence.topology_class,
            credit_weight=credit_weight,
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
        return experience

    def _transitions_from_steps(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> list[SkillTransition]:
        transitions: list[SkillTransition] = []
        observation_history: list[str] = []
        for index, step in enumerate(tool_steps):
            state = str(_sanitize_public_value(evidence.task_description, evidence))
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
                    policy_token_budget=step.policy_token_budget,
                    selection_probability=step.selection_probability,
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
        """Build one replay trajectory per Skill and episode.

        Activation ids remain attached to transitions for exact attribution,
        but repeated activations in one episode are correlated segments of the
        same trajectory and must not count as independent evolution samples.
        """

        sanitized_steps = [_sanitize_skill_step(step, evidence) for step in tool_steps]
        all_transitions = self._transitions_from_steps(
            evidence=evidence,
            tool_steps=sanitized_steps,
        )
        grouped: dict[str, tuple[list[SkillStep], list[SkillTransition]]] = {}
        for step, transition in zip(sanitized_steps, all_transitions, strict=True):
            skill_id = step.skill_id
            activation_id = step.activation_id
            valid_attribution = (
                bool(skill_id)
                and bool(activation_id)
                and (valid_skill_ids is None or skill_id in valid_skill_ids)
            )
            if not valid_attribution:
                continue
            grouped_steps, grouped_transitions = grouped.setdefault(
                skill_id,
                ([], []),
            )
            grouped_steps.append(step)
            grouped_transitions.append(transition)
        for _, grouped_transitions in grouped.values():
            for transition in grouped_transitions:
                transition.done = False
            grouped_transitions[-1].done = True
        activation_counts = {
            skill_id: len(
                {
                    transition.activation_id
                    for transition in transitions
                    if transition.activation_id
                }
            )
            for skill_id, (_, transitions) in grouped.items()
        }
        total_activations = max(sum(activation_counts.values()), 1)
        return {
            skill_id: self._experience_from_episode(
                evidence=evidence,
                tool_steps=steps,
                reward=reward,
                baseline=baseline,
                skill_ids=[skill_id],
                success=success,
                transitions=transitions,
                credit_weight=activation_counts.get(skill_id, 1) / total_activations,
            )
            for skill_id, (steps, transitions) in grouped.items()
        }

    @staticmethod
    def _coalesce_legacy_experiences(
        experiences: list[SkillExperience],
        *,
        skill_id: str,
    ) -> list[SkillExperience]:
        """Merge pre-migration activation records into trajectory records."""

        by_session: dict[str, list[SkillExperience]] = defaultdict(list)
        for experience in experiences:
            by_session[experience.session_id].append(experience)
        merged: list[SkillExperience] = []
        for session_id, session_items in by_session.items():
            if len(session_items) == 1:
                merged.append(session_items[0])
                continue
            first = session_items[0]
            transitions = [
                transition for item in session_items for transition in item.transitions
            ]
            for transition in transitions:
                transition.done = False
            if transitions:
                transitions[-1].done = True
            merged.append(
                first.model_copy(
                    update={
                        "experience_id": _stable_id(
                            session_id,
                            skill_id,
                            [item.experience_id for item in session_items],
                            prefix="exp",
                        ),
                        "skill_ids": [skill_id],
                        "transitions": transitions,
                        "step_count": len(transitions),
                        "used_for_evolution": all(
                            item.used_for_evolution for item in session_items
                        ),
                    }
                )
            )
        return merged

    def _evolution_batch(
        self,
        state,
        parent: ProceduralSkill | None,
    ) -> list[SkillExperience]:
        if parent is None:
            return []
        pool = [
            exp
            for exp in state.experiences
            if parent.skill_id in exp.skill_ids and not exp.used_for_evolution
        ]
        pool = self._coalesce_legacy_experiences(pool, skill_id=parent.skill_id)
        batch_size = max(self.evolution_threshold, 2)
        if len(pool) <= batch_size:
            return list(pool)
        ordered = sorted(pool, key=lambda exp: exp.reward)
        quantile_indices = [
            round(index * (len(ordered) - 1) / (batch_size - 1))
            for index in range(batch_size)
        ]
        quantile_candidates = [ordered[index] for index in quantile_indices]
        selected: list[SkillExperience] = []
        seen_ids: set[str] = set()
        seen_strata: set[tuple[str, str, bool | None]] = set()
        for exp in quantile_candidates:
            stratum = (
                exp.scenario,
                exp.topology_class,
                exp.ground_truth_is_anomaly,
            )
            if stratum in seen_strata:
                continue
            selected.append(exp)
            seen_ids.add(exp.experience_id)
            seen_strata.add(stratum)
        for exp in [*quantile_candidates, *ordered]:
            if len(selected) >= batch_size:
                break
            if exp.experience_id in seen_ids:
                continue
            selected.append(exp)
            seen_ids.add(exp.experience_id)
        return selected

    def _next_evolution_parent(self, state) -> ProceduralSkill | None:
        buffered: list[tuple[ProceduralSkill, int]] = []
        for skill in state.skills.values():
            if skill.status == "retired":
                continue
            available = self._evolution_batch(state, skill)
            if available:
                buffered.append((skill, len(available)))
        if not buffered:
            return None
        required = max(self.evolution_threshold, 2)
        ready = [item for item in buffered if item[1] >= required]
        candidates = ready or buffered
        skill, _ = max(
            candidates,
            key=lambda item: (
                item[1] >= required,
                self._skill_evolution_priority(state, item[0]),
                item[1],
                item[0].skill_id,
            ),
        )
        return skill

    @staticmethod
    def _tool_steps_from_experiences(
        experiences: list[SkillExperience],
    ) -> list[SkillStep]:
        steps: list[SkillStep] = []
        for experience in experiences:
            for transition in experience.transitions:
                steps.append(
                    SkillStep(
                        order=len(steps) + 1,
                        action=transition.action,
                        skill_id=transition.skill_id,
                        tool_name=transition.tool_name,
                        arguments_hint=transition.arguments_hint,
                        observation_summary=transition.observation_summary,
                        status=transition.status,
                        rationale="Observed Skill-Pro trajectory transition.",
                        policy_state=transition.state,
                        policy_context=transition.policy_context,
                        policy_token_budget=transition.policy_token_budget,
                        selection_probability=transition.selection_probability,
                        activation_id=transition.activation_id,
                    )
                )
        return steps

    def _verification_batch(
        self,
        state,
        *,
        generation_samples: list[SkillExperience],
    ) -> list[SkillExperience]:
        """Reserve a deterministic trajectory-level holdout."""
        if len(generation_samples) < 2:
            return []
        if len({sample.session_id for sample in generation_samples}) != len(
            generation_samples
        ):
            raise ValueError("verification batch contains duplicate session ids")
        holdout_count = min(self.holdout_size, len(generation_samples) - 1)
        ordered = sorted(
            generation_samples,
            key=lambda item: (
                abs(item.advantage),
                item.advantage > 0,
                hashlib.sha256(item.experience_id.encode("utf-8")).hexdigest(),
            ),
        )
        del state
        # Holdout advantages remain anchored to the behavior-time baseline.
        # Recomputing them with the current EMA leaks the holdout reward and can
        # change the sign of the learning signal after collection.
        return ordered[-holdout_count:]

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
                "candidate_j_score": candidate_score,
                "parent_j_score": baseline_score,
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
        parent_total = 0.0
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
            advantage = exp.advantage
            transition_count = max(len(exp.transitions), 1)
            per_step = advantage / max(exp.step_count, transition_count, 1)
            surrogate = min(raw_ratio * per_step, clipped_ratio * per_step)
            total += surrogate * transition_count
            parent_total += per_step * transition_count
            steps += transition_count
            candidate_alignment_total += candidate_alignment * transition_count
            baseline_alignment_total += baseline_alignment * transition_count
        candidate_j_score = total / max(steps, 1)
        parent_j_score = parent_total / max(steps, 1)
        return {
            "j_score": candidate_j_score - parent_j_score,
            "candidate_j_score": candidate_j_score,
            "parent_j_score": parent_j_score,
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
        parent_total = 0.0
        steps = 0
        candidate_logprob_total = 0.0
        baseline_logprob_total = 0.0
        verification_error = replay_scores.error
        for experience in samples:
            transition_count = len(experience.transitions)
            if transition_count <= 0:
                continue
            advantage = experience.advantage
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
                parent_total += step_advantage
                candidate_logprob_total += score.candidate_logprob
                baseline_logprob_total += score.baseline_logprob
                steps += 1
        if steps <= 0:
            verification_error = (
                verification_error or "logprob scorer returned no steps"
            )
        candidate_j_score = total / max(steps, 1)
        parent_j_score = parent_total / max(steps, 1)
        return {
            "j_score": candidate_j_score - parent_j_score,
            "candidate_j_score": candidate_j_score,
            "parent_j_score": parent_j_score,
            "candidate_alignment": candidate_logprob_total / max(steps, 1),
            "baseline_alignment": baseline_logprob_total / max(steps, 1),
            "verification_method": "policy_logprob",
            "verification_error": verification_error,
        }

    def _update_baseline(self, state, scenario: str, reward: float) -> None:
        old = state.baselines.get(scenario, 0.0)
        state.baselines[scenario] = (
            1 - self.baseline_ema_alpha
        ) * old + self.baseline_ema_alpha * reward

    def _skill_effective_score(self, skill: ProceduralSkill) -> float:
        prior = skill.prior_score or max(skill.score, 0.0)
        if skill.frequency <= 0:
            return prior
        prior_weight = 2.0 if _is_seed_skill(skill) else 1.0
        return (prior_weight * prior + skill.total_gain) / (
            prior_weight + skill.frequency
        )

    def _maintain(self, state) -> None:
        logs: list[dict[str, Any]] = []
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

        probation_support = max(3, self.holdout_size)
        for skill in active:
            if skill.status == "probationary" and skill.frequency >= probation_support:
                probation_lcb = skill.avg_gain - 0.10 * math.sqrt(
                    math.log1p(max(current_iteration, 1)) / max(skill.frequency, 1)
                )
                if (
                    probation_lcb > self.acceptance_margin
                    and skill.success_count >= self.min_positive_advantage
                ):
                    skill.status = "validated"
                    logs.append(
                        {
                            "stage": "promote probationary skill",
                            "skill_id": skill.skill_id,
                            "frequency": skill.frequency,
                            "avg_gain": skill.avg_gain,
                            "gain_lcb": probation_lcb,
                        }
                    )
                    parent = state.skills.get(skill.parent_id)
                    if (
                        parent is not None
                        and parent.origin == "learned"
                        and parent.status != "retired"
                    ):
                        retire(
                            parent,
                            "superseded by promoted child",
                            duplicate_of=skill.skill_id,
                        )
                elif skill.avg_gain < -self.acceptance_margin:
                    retire(skill, "probationary online rollback")
            elif (
                skill.status == "validated"
                and skill.origin == "learned"
                and skill.frequency >= probation_support
                and skill.avg_gain < -self.acceptance_margin
            ):
                retire(skill, "validated online rollback")

        for skill in [item for item in active if item.status != "retired"]:
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
