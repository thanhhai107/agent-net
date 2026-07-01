"""Skill-Pro procedural memory service.

The module stores reusable diagnostic procedures as Skill-MDP records and uses
a non-parametric PPO gate before accepting a new or revised skill.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import (
    EvaluationEvidence,
    MemoryQuery,
    PPOGateDecision,
    ProceduralSkill,
    SemanticGradient,
    SkillRetrieval,
    SkillStep,
)
from agent.memory.store import SkillMemoryStore
from agent.llm.model_factory import load_model


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
    return [
        step.model_dump(exclude_none=True, mode="json")
        for step in tool_steps[:12]
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


class ProceduralMemoryModule:
    def __init__(
        self,
        *,
        bank_id: str = "default",
        llm_backend: str | None = None,
        model: str | None = None,
        store_path: Path | None = None,
    ) -> None:
        self.bank_id = bank_id
        self.llm_backend = llm_backend
        self.model = model
        self.store = SkillMemoryStore(
            bank_id=bank_id,
            root=store_path.parent if store_path else None,
        )

    def clear(self) -> None:
        self.store.clear()

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
        for skill in state.skills.values():
            if skill.status == "retired":
                continue
            reasons: list[str] = []
            score = skill.score
            for label, values in (
                ("protocol", skill.protocols),
                ("service", skill.services),
                ("symptom", skill.symptoms),
                ("tool", skill.tools),
            ):
                overlap = set(values) & set(getattr(query, f"{label}s", []) if label != "tool" else query.tools)
                if overlap:
                    score += 0.15 * len(overlap)
                    reasons.append(f"{label}:{','.join(sorted(overlap))}")
            for token in skill.activation_condition.lower().split():
                if len(token) > 3 and token in query_text:
                    score += 0.01
            if score > 0:
                scored.append(SkillRetrieval(memory=skill, score=score, reasons=reasons))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: query.top_k]

    def format_context(self, retrieved: list[SkillRetrieval]) -> str:
        if not retrieved:
            return ""
        blocks = [
            "Retrieved Skill-Pro procedural skills. Use them as reusable diagnostic policies, not as ground truth."
        ]
        for item in retrieved:
            skill = item.skill
            steps = "; ".join(
                f"{step.order}. {step.action}" for step in skill.execution_steps[:6]
            )
            blocks.append(
                "\n".join(
                    [
                        f"- Skill {skill.skill_id} ({skill.title}) score={item.score:.3f}",
                        f"  Activation: {skill.activation_condition}",
                        f"  Steps: {steps}",
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
    ) -> ProceduralSkill:
        if not tool_steps:
            raise ValueError("Skill-Pro requires at least one observed execution step.")
        attrs = infer_memory_attributes(
            evidence.task_description,
            scenario=evidence.scenario,
            topology_class=evidence.topology_class,
            tools=[step.tool_name for step in tool_steps if step.tool_name],
        )
        topic = _skill_topic(
            evidence,
            attrs.protocols,
            attrs.services,
            attrs.symptoms,
        )
        skill_id = _stable_id(
            evidence.scenario,
            attrs.protocols,
            attrs.services,
            attrs.symptoms,
            attrs.tools,
            prefix="skill",
        )
        critique = self.semantic_gradient(evidence=evidence, tool_steps=tool_steps)
        termination = (
            "Stop when anomaly status, faulty devices, and root-cause class are supported "
            "by at least two independent observations, or when max diagnostic budget is reached."
        )
        if critique.proposed_update:
            termination += f" Semantic update: {critique.proposed_update[:240]}"
        return ProceduralSkill(
            skill_id=skill_id,
            title=f"Procedure for {topic}",
            activation_condition=(
                f"Use when task resembles {evidence.scenario or 'the current scenario'} "
                f"with symptoms: {', '.join(attrs.symptoms) or evidence.task_description[:120]}."
            ),
            execution_steps=tool_steps[:10],
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
            semantic_gradients=[critique],
        )

    def semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        llm_gradient = self._llm_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )
        if llm_gradient is not None:
            return llm_gradient
        return self._deterministic_semantic_gradient(
            evidence=evidence,
            tool_steps=tool_steps,
        )

    def _deterministic_semantic_gradient(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> SemanticGradient:
        if evidence.success:
            critique = "Successful trajectory: preserve the evidence order and termination rule."
            update = "Promote or reinforce this procedure if it improves step/tool efficiency."
        else:
            critique = (
                "Failed trajectory: revise activation or execution steps to require stronger "
                "evidence before localization/RCA."
            )
            update = "Store only as candidate unless PPO gate beats the existing/default policy."
        if not tool_steps:
            critique += " Trace contained no usable diagnosis tool calls."
        return SemanticGradient(
            source_session_id=evidence.session_id,
            critique=critique,
            proposed_update=update,
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
            "You are the Skill-Pro semantic-gradient critic for NIKA network "
            "diagnosis. Produce a concise critique and proposed procedural "
            "skill update from public task context, execution steps, and "
            "numeric evaluation metrics. Do not name hidden root causes or "
            "faulty devices.\n\n"
            f"Evaluation evidence:\n{json.dumps(public_evidence, indent=2, ensure_ascii=False)}\n\n"
            f"Observed execution steps:\n{json.dumps(_skill_steps_summary(tool_steps), indent=2, ensure_ascii=False)}\n\n"
            "Return a SemanticGradient with the same source_session_id."
        )
        try:
            critic = load_model(self.llm_backend, self.model).with_structured_output(
                SemanticGradient
            )
            gradient = critic.invoke(prompt)
            if not isinstance(gradient, SemanticGradient):
                gradient = SemanticGradient.model_validate(gradient)
            if gradient.source_session_id != evidence.session_id:
                gradient.source_session_id = evidence.session_id
            gradient.gradient_source = "llm"
            return gradient
        except Exception:
            return None

    def ppo_gate(
        self,
        *,
        candidate: ProceduralSkill,
        evidence: EvaluationEvidence,
    ) -> PPOGateDecision:
        state = self.store.load()
        candidate_score = candidate.score
        baseline: ProceduralSkill | None = None
        for skill in state.skills.values():
            if skill.skill_id == candidate.skill_id:
                baseline = skill
                break
            if set(skill.protocols) & set(candidate.protocols) or set(skill.symptoms) & set(candidate.symptoms):
                if baseline is None or skill.score > baseline.score:
                    baseline = skill
        baseline_score = baseline.score if baseline else 0.0
        margin = 0.03
        accepted = candidate_score >= baseline_score + margin or (
            evidence.success and baseline is None
        )
        reason = (
            "candidate improves non-parametric PPO score"
            if accepted
            else "candidate did not beat existing/default policy"
        )
        return PPOGateDecision(
            accepted=accepted,
            reason=reason,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            replaced_skill_id=baseline.skill_id if baseline and accepted else None,
        )

    def learn_from_episode(
        self,
        *,
        evidence: EvaluationEvidence,
        tool_steps: list[SkillStep],
    ) -> dict[str, Any]:
        self.store.record_episode(evidence)
        if not tool_steps:
            return {
                "status": "rejected",
                "reason": "Skill-Pro requires at least one observed execution step.",
                "skill_id": "",
                "decision": None,
                "skills": len(self.store.load().skills),
            }
        candidate = self.propose_skill(evidence=evidence, tool_steps=tool_steps)
        decision = self.ppo_gate(candidate=candidate, evidence=evidence)
        gradient_source = (
            candidate.semantic_gradients[-1].gradient_source
            if candidate.semantic_gradients
            else "deterministic"
        )
        state = self.store.load()
        if decision.accepted:
            old = state.skills.get(candidate.skill_id)
            if old is not None:
                candidate.reuse_count = old.reuse_count
                candidate.success_count += old.success_count
                candidate.failure_count += old.failure_count
                candidate.source_sessions = sorted(set(old.source_sessions + candidate.source_sessions))
                candidate.semantic_gradients = old.semantic_gradients + candidate.semantic_gradients
            state.skills[candidate.skill_id] = candidate
        state.ppo_decisions.append(decision)
        self._maintain(state)
        self.store.save(state)
        return {
            "status": "accepted" if decision.accepted else "rejected",
            "skill_id": candidate.skill_id,
            "semantic_gradient_source": gradient_source,
            "decision": decision.model_dump(),
            "skills": len(state.skills),
        }

    def _maintain(self, state) -> None:
        seen_hashes: dict[str, str] = {}
        for skill in list(state.skills.values()):
            total = skill.success_count + skill.failure_count
            success_rate = skill.success_count / max(total, 1)
            skill.score = max(skill.score, success_rate * 0.7 + min(skill.reuse_count, 10) * 0.03)
            digest = skill.content_hash()
            duplicate_of = seen_hashes.get(digest)
            if duplicate_of:
                skill.status = "retired"
            else:
                seen_hashes[digest] = skill.skill_id
            if total >= 3 and success_rate < 0.25:
                skill.status = "retired"
