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
from pathlib import Path
from typing import Any

from agent.llm.model_factory import load_model
from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import (
    EvaluationEvidence,
    MemoryQuery,
    PPOGateDecision,
    ProceduralSkill,
    SemanticGradient,
    SkillComponentGradient,
    SkillExperience,
    SkillRetrieval,
    SkillStep,
    SkillTransition,
    utc_now,
)
from agent.memory.store import SkillMemoryStore

DEFAULT_POOL_SIZE = 32
EXPERIENCE_POOL_SIZE = 1000
GOLDEN_POOL_SIZE = 20
PPO_EPSILON = 0.2
BASELINE_EMA_ALPHA = 0.1


def _stable_id(*parts: Any, prefix: str) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def _metric_success(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("detection_score") or 0) >= 1.0
        and float(metrics.get("localization_accuracy") or 0) >= 0.5
        and float(metrics.get("rca_accuracy") or 0) >= 0.5
    )


def _evidence_score(evidence: EvaluationEvidence) -> float:
    accuracy = (
        float(evidence.metrics.get("detection_score") or 0) * 0.2
        + float(evidence.metrics.get("localization_accuracy") or 0) * 0.3
        + float(evidence.metrics.get("rca_accuracy") or 0) * 0.5
    )
    step_penalty = min(evidence.steps or 0, 100) / 250.0
    tool_penalty = min(evidence.tool_calls or 0, 200) / 500.0
    return max(0.0, accuracy - step_penalty - tool_penalty)


def _skill_steps_summary(tool_steps: list[SkillStep]) -> list[dict[str, Any]]:
    return [step.model_dump(exclude_none=True, mode="json") for step in tool_steps[:12]]


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _skill_base_id(skill_id: str) -> str:
    return re.sub(r"_v\d+(?:_[a-f0-9]{6})?$", "", skill_id)


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
        return [
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
            reasons: list[str] = []
            score = max(skill.score, skill.avg_gain)
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
            for token in skill.activation_condition.lower().split():
                if len(token) > 3 and token in query_text:
                    score += 0.01
            score += self._lcb_bonus(skill, total_maturity)
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
    ) -> SkillRetrieval | None:
        top_k = top_k or query.top_k
        candidates = self.retrieve(
            query=query.model_copy(update={"top_k": max(top_k, 1)}),
            session_id=session_id,
        )
        if not candidates:
            return None
        selected = candidates[0]
        if selected.skill.maturity >= 3 and self._lcb(selected.skill) < min_lcb:
            return None
        if record_reuse:
            state = self.store.load()
            stored = state.skills.get(selected.skill.skill_id)
            if stored is not None:
                stored.reuse_count += 1
                stored.updated_at = utc_now()
                state.skills[stored.skill_id] = stored
                self.store.save(state)
                selected.memory = stored
        return selected

    def format_context(self, retrieved: list[SkillRetrieval]) -> str:
        if not retrieved:
            return ""
        blocks = [
            "Retrieved Skill-Pro Skill-MDP procedures. Treat them as reusable diagnostic policies, not as ground truth."
        ]
        for index, item in enumerate(retrieved):
            skill = item.skill
            label = "ACTIVE" if index == 0 else "CANDIDATE"
            blocks.append(
                "\n".join(
                    [
                        f"- {label} Skill {skill.skill_id} ({skill.title}) score={item.score:.3f}",
                        f"  Activation / Initiation: {skill.activation_condition}",
                        "  Policy:",
                        *[f"    {step.order}. {step.action}" for step in skill.execution_steps[:6]],
                        f"  Termination: {skill.termination_condition}",
                    ]
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
    ) -> ProceduralSkill:
        if not tool_steps:
            raise ValueError("Skill-Pro requires at least one observed execution step.")
        attrs = infer_memory_attributes(
            evidence.task_description,
            scenario=evidence.scenario,
            topology_class=evidence.topology_class,
            tools=[step.tool_name for step in tool_steps if step.tool_name],
        )
        topic = _skill_topic(evidence, attrs.protocols, attrs.services, attrs.symptoms)
        critique = self.semantic_gradient(evidence=evidence, tool_steps=tool_steps)
        if parent is None:
            skill_id = _stable_id(
                evidence.scenario,
                attrs.protocols,
                attrs.services,
                attrs.symptoms,
                attrs.tools,
                prefix="skill",
            )
            title = f"Procedure for {topic}"
            activation = (
                f"Use when task resembles {evidence.scenario or 'the current scenario'} "
                f"with symptoms: {', '.join(attrs.symptoms) or evidence.task_description[:120]}."
            )
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
            activation = critique.component_update.initiation or parent.activation_condition
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
        return ProceduralSkill(
            skill_id=skill_id,
            title=title,
            activation_condition=activation,
            execution_steps=steps[:10],
            termination_condition=termination,
            source_sessions=[evidence.session_id],
            protocols=attrs.protocols,
            services=attrs.services,
            symptoms=attrs.symptoms,
            tools=attrs.tools,
            status="validated" if evidence.success else "candidate",
            success_count=1 if evidence.success else 0,
            failure_count=0 if evidence.success else 1,
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
        llm_gradient = self._llm_semantic_gradient(evidence=evidence, tool_steps=tool_steps)
        if llm_gradient is not None:
            return llm_gradient
        return self._deterministic_semantic_gradient(evidence=evidence, tool_steps=tool_steps)

    def _deterministic_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        if evidence.success:
            critique = "Successful trajectory: preserve evidence order and termination rule."
            update = "Promote or reinforce this procedure if it improves step/tool efficiency."
            component = SkillComponentGradient(
                policy=[step.action for step in tool_steps[:6]],
                termination="Terminate after direct evidence supports detection, localization, and RCA.",
                is_related=True,
            )
        else:
            critique = (
                "Failed trajectory: revise initiation or policy to require stronger "
                "evidence before localization/RCA."
            )
            update = "Store only as candidate unless PPO gate beats the existing/default policy."
            component = SkillComponentGradient(
                policy=[
                    "Collect broad anomaly evidence before narrowing the faulty device.",
                    "Verify the suspected root cause with an independent command.",
                ],
                termination="Do not terminate until diagnosis has at least two independent observations.",
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
    ) -> SemanticGradient | None:
        if not self.llm_backend or not self.model:
            return None
        public_evidence = evidence.model_dump(
            exclude={"root_cause", "faulty_devices"},
            mode="json",
        )
        prompt = (
            "You are the Skill-Pro semantic-gradient critic for NIKA network diagnosis. "
            "Generate component-level updates for a Skill-MDP option. Do not name hidden "
            "root causes or faulty devices.\n\n"
            f"Evaluation evidence:\n{json.dumps(public_evidence, indent=2, ensure_ascii=False)}\n\n"
            f"Observed execution steps:\n{json.dumps(_skill_steps_summary(tool_steps), indent=2, ensure_ascii=False)}\n\n"
            "Return a SemanticGradient. component_update.initiation updates the Initiation, "
            "component_update.policy updates the Policy steps, and component_update.termination "
            "updates the Termination rule. Use the same source_session_id."
        )
        try:
            critic = load_model(self.llm_backend, self.model).with_structured_output(SemanticGradient)
            gradient = critic.invoke(prompt)
            if not isinstance(gradient, SemanticGradient):
                gradient = SemanticGradient.model_validate(gradient)
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
            return gradient
        except Exception:
            return None

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
        candidate_score = candidate.score
        baseline_score = baseline.score if baseline else 0.0
        sample_batch = samples or [
            SkillExperience(
                experience_id=_stable_id(evidence.session_id, "gate", prefix="exp"),
                session_id=evidence.session_id,
                reward=_evidence_score(evidence),
                baseline=baseline_score,
                advantage=_evidence_score(evidence) - baseline_score,
                success=evidence.success,
            )
        ]
        j_score = self._ppo_surrogate(candidate, baseline=baseline, samples=sample_batch)
        margin = 0.03
        accepted = (
            evidence.success
            and baseline is None
            and candidate_score > 0
        ) or (
            candidate_score >= baseline_score + margin
            and j_score > -margin
        )
        reason = (
            "candidate passed Skill-Pro PPO gate"
            if accepted
            else "candidate failed Skill-Pro PPO gate"
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
        if not tool_steps:
            if not any(item.session_id == evidence.session_id for item in state.episodes):
                state.episodes.append(evidence)
                self.store.save(state)
            return {
                "status": "rejected",
                "reason": "Skill-Pro requires at least one observed execution step.",
                "skill_id": "",
                "decision": None,
                "skills": len(state.skills),
            }

        parent_item = self._select_parent_for_evidence(evidence=evidence, tool_steps=tool_steps)
        parent = parent_item.skill if parent_item is not None else None
        reward = _evidence_score(evidence)
        baseline_value = state.baselines.get(evidence.scenario or "default", 0.0)
        experience = self._experience_from_episode(
            evidence=evidence,
            tool_steps=tool_steps,
            reward=reward,
            baseline=baseline_value,
            skill_ids=[parent.skill_id] if parent else [],
        )

        if not any(item.session_id == evidence.session_id for item in state.episodes):
            state.episodes.append(evidence)
        if not any(item.experience_id == experience.experience_id for item in state.experiences):
            state.experiences.append(experience)
            state.experiences = state.experiences[-EXPERIENCE_POOL_SIZE:]
        self._update_golden_pool(state, experience)
        self._update_baseline(state, evidence.scenario or "default", reward)
        state.iteration += 1
        for skill in state.skills.values():
            skill.increment_maturity()
        if parent is not None and parent.skill_id in state.skills:
            total_calls = max(1, len(experience.skill_ids))
            state.skills[parent.skill_id].update_stats(
                reward=reward,
                baseline=baseline_value,
                total_skill_calls=total_calls,
                skill_call_count=1,
            )

        samples = self._evolution_batch(state, parent)
        best_decision: PPOGateDecision | None = None
        best_candidate: ProceduralSkill | None = None
        candidate_type = "REFINE" if parent is not None else "NEW"
        for index in range(self.best_of_n):
            candidate = self.propose_skill(
                evidence=evidence,
                tool_steps=tool_steps,
                parent=parent,
                candidate_index=index,
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
        state.ppo_decisions.append(best_decision)
        state.evolution_log.append(
            {
                "iteration": state.iteration,
                "parent": parent.skill_id if parent else "",
                "candidate": best_candidate.skill_id,
                "action": "accepted" if best_decision.accepted else "rejected",
                "j_score": best_decision.j_score,
                "sample_count": best_decision.sample_count,
                "best_of_n": self.best_of_n,
            }
        )
        self._maintain(state)
        self.store.save(state)
        return {
            "status": "accepted" if best_decision.accepted else "rejected",
            "skill_id": best_candidate.skill_id,
            "semantic_gradient_source": gradient_source,
            "decision": best_decision.model_dump(),
            "skills": len(state.skills),
            "experience_id": experience.experience_id,
            "method": "Skill-Pro",
        }

    def _select_parent_for_evidence(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SkillRetrieval | None:
        attrs = infer_memory_attributes(
            evidence.task_description,
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
    ) -> SkillExperience:
        transitions = [
            SkillTransition(
                state=evidence.task_description,
                action=step.action,
                skill_id=skill_ids[0] if skill_ids else "",
                tool_name=step.tool_name,
                arguments_hint=step.arguments_hint,
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
            total_added_tokens=0,
            success=evidence.success,
        )

    def _evolution_batch(
        self,
        state,
        parent: ProceduralSkill | None,
    ) -> list[SkillExperience]:
        if parent is None:
            pool = state.experiences[-self.evolution_threshold :]
        else:
            pool = [
                exp
                for exp in state.experiences
                if parent.skill_id in exp.skill_ids
            ]
            if len(pool) < self.evolution_threshold:
                pool = pool + [
                    exp
                    for exp in state.golden_experiences
                    if exp.experience_id not in {item.experience_id for item in pool}
                ]
        if len(pool) <= self.evolution_threshold:
            return list(pool)
        ordered = sorted(pool, key=lambda exp: exp.reward)
        half = max(1, self.evolution_threshold // 2)
        batch = ordered[:half] + ordered[-half:]
        seen: dict[str, SkillExperience] = {}
        for exp in batch:
            seen[exp.experience_id] = exp
        return list(seen.values())

    def _ppo_surrogate(
        self,
        candidate: ProceduralSkill,
        *,
        baseline: ProceduralSkill | None,
        samples: list[SkillExperience],
    ) -> float:
        if not samples:
            return candidate.score - (baseline.score if baseline else 0.0)
        candidate_text = candidate.format_for_llm()
        baseline_text = baseline.format_for_llm() if baseline else candidate_text
        similarity = _jaccard(candidate_text, baseline_text)
        raw_ratio = 0.5 + similarity
        clipped_ratio = _clamp(raw_ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
        total = 0.0
        steps = 0
        for exp in samples:
            advantage = exp.reward - exp.baseline
            per_step = advantage / max(len(exp.transitions), 1)
            surrogate = min(raw_ratio * per_step, clipped_ratio * per_step)
            total += surrogate * max(len(exp.transitions), 1)
            steps += max(len(exp.transitions), 1)
        return total / max(steps, 1)

    def _update_baseline(self, state, scenario: str, reward: float) -> None:
        old = state.baselines.get(scenario)
        state.baselines[scenario] = reward if old is None else (1 - BASELINE_EMA_ALPHA) * old + BASELINE_EMA_ALPHA * reward

    def _update_golden_pool(self, state, experience: SkillExperience) -> None:
        if not experience.transitions:
            return
        pool = {item.experience_id: item for item in state.golden_experiences}
        pool[experience.experience_id] = experience
        state.golden_experiences = sorted(pool.values(), key=lambda item: item.reward, reverse=True)[:GOLDEN_POOL_SIZE]

    def _lcb(self, skill: ProceduralSkill) -> float:
        state = self.store.load()
        t = max([item.maturity for item in state.skills.values()] or [1])
        n = max(skill.frequency, 1)
        return skill.avg_gain - 0.2 * math.sqrt(math.log1p(t) / n)

    def _lcb_bonus(self, skill: ProceduralSkill, total_maturity: int) -> float:
        n = max(skill.frequency, 1)
        return max(-0.2, skill.avg_gain - 0.2 * math.sqrt(math.log1p(max(total_maturity, 1)) / n))

    def _maintain(self, state) -> None:
        total_frequency = sum(skill.frequency for skill in state.skills.values())
        seen_hashes: dict[str, str] = {}
        active = [skill for skill in state.skills.values() if skill.status != "retired"]
        logs: list[dict[str, Any]] = []
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
            skill.score = max(skill.score, skill.maintenance_score(total_frequency))
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
