"""Skill-Pro procedural skill schemas for NIKA diagnosis agents."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.module_config import module_defaults


_DEFAULTS = module_defaults().procedural_memory
DEFAULT_MANUAL_RETRIEVAL_TOP_K = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProceduralMemoryAttributes(BaseModel):
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    task_stages: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class ProceduralMemoryQuery(BaseModel):
    text: str
    scenario: str = ""
    topology_class: str = ""
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    task_stage: str = "diagnosis"
    tools: list[str] = Field(default_factory=list)
    top_k: int = DEFAULT_MANUAL_RETRIEVAL_TOP_K
    token_budget: int = _DEFAULTS.token_budget


class SkillStep(BaseModel):
    order: int
    action: str
    skill_id: str = ""
    tool_name: str = ""
    arguments_hint: dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = ""
    status: Literal["success", "error", "unknown"] = "unknown"
    rationale: str = ""
    policy_state: str = ""
    policy_context: str = ""
    activation_id: str = ""


class SkillComponentGradient(BaseModel):
    initiation: str = ""
    policy: list[str] = Field(default_factory=list)
    termination: str = ""
    is_related: bool = True


class SemanticGradient(BaseModel):
    source_session_id: str
    critique: str
    proposed_update: str
    component_update: SkillComponentGradient = Field(
        default_factory=SkillComponentGradient
    )
    gradient_source: Literal["llm", "deterministic"] = "deterministic"
    created_at: str = Field(default_factory=utc_now)


class SemanticGradientDraft(BaseModel):
    """Small structured critic payload used for bounded learning LLM calls."""

    source_session_id: str = ""
    critique: str = ""
    proposed_update: str = ""
    initiation: str = ""
    policy: list[str] = Field(default_factory=list)
    termination: str = ""
    is_related: bool = True


class SkillCandidateDraft(BaseModel):
    """One independently sampled Skill-Pro evolution candidate."""

    title: str = ""
    initiation: str = ""
    policy: list[str] = Field(default_factory=list)
    termination: str = ""


class SkillSelectionDraft(BaseModel):
    """Strict output contract for the online Skill-MDP selector."""

    skill_id: str = ""
    reason: str = ""


class EvaluationEvidence(BaseModel):
    session_id: str
    task_description: str = ""
    scenario: str = ""
    topology_class: str = ""
    root_cause: list[str] = Field(default_factory=list)
    faulty_devices: list[str] = Field(default_factory=list)
    ground_truth_is_anomaly: bool | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    steps: int = 0
    tool_calls: int = 0
    success: bool = False


class SkillTransition(BaseModel):
    state: str = ""
    action: str = ""
    skill_id: str = ""
    tool_name: str = ""
    arguments_hint: dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = ""
    status: Literal["success", "error", "unknown"] = "unknown"
    done: bool = False
    policy_context: str = ""
    activation_id: str = ""


class SkillExperience(BaseModel):
    experience_id: str
    session_id: str
    reward: float
    baseline: float = 0.0
    advantage: float = 0.0
    skill_ids: list[str] = Field(default_factory=list)
    trajectory: str = ""
    scenario: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    transitions: list[SkillTransition] = Field(default_factory=list)
    step_count: int = 0
    total_added_tokens: int = 0
    used_for_evolution: bool = False
    success: bool = False
    ground_truth_is_anomaly: bool | None = None
    created_at: str = Field(default_factory=utc_now)


class ProceduralSkill(BaseModel):
    skill_id: str
    title: str
    activation_condition: str
    execution_steps: list[SkillStep] = Field(min_length=1)
    termination_condition: str
    source_sessions: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    status: Literal["candidate", "probationary", "validated", "retired"] = "candidate"
    reuse_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    score: float = 0.0
    prior_score: float = 0.0
    frequency: int = 0
    total_gain: float = 0.0
    avg_gain: float = 0.0
    maturity: int = 0
    parent_id: str = ""
    version: int = 0
    last_evolved_iteration: int = 0
    semantic_gradients: list[SemanticGradient] = Field(default_factory=list)
    origin: Literal["learned", "generic_seed"] = "learned"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def content_hash(self) -> str:
        payload = {
            "activation_condition": self.activation_condition,
            "execution_steps": [
                {"action": step.action, "tool_name": step.tool_name}
                for step in self.execution_steps
            ],
            "termination_condition": self.termination_condition,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def update_stats(
        self,
        *,
        reward: float,
        baseline: float,
        total_skill_calls: int,
        skill_call_count: int = 1,
    ) -> None:
        advantage = reward - baseline
        per_call_gain = advantage / max(total_skill_calls, 1)
        self.total_gain += skill_call_count * per_call_gain
        self.frequency += skill_call_count
        self.avg_gain = self.total_gain / max(self.frequency, 1)
        if advantage > 0:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.updated_at = utc_now()

    def increment_maturity(self) -> None:
        self.maturity += 1
        self.updated_at = utc_now()

    def format_for_llm(self) -> str:
        steps = "\n".join(f"- {step.action}" for step in self.execution_steps)
        return (
            f"Skill Name: {self.skill_id}\n"
            f"Initiation (When to use): {self.activation_condition}\n"
            f"Strategy Steps:\n{steps}\n"
            "Option Termination (when to return control to the skill selector): "
            f"{self.termination_condition}\n"
            "Option termination is not task completion and never authorizes a "
            "diagnosis submission."
        )


class SkillRetrieval(BaseModel):
    skill: ProceduralSkill
    score: float
    reasons: list[str] = Field(default_factory=list)


class PPOGateDecision(BaseModel):
    accepted: bool
    reason: str
    candidate_score: float
    baseline_score: float
    replaced_skill_id: str | None = None
    candidate_skill_id: str = ""
    parent_skill_id: str = ""
    j_score: float = 0.0
    candidate_alignment: float = 0.0
    baseline_alignment: float = 0.0
    sample_count: int = 0
    best_of_n: int = 1
    candidate_type: Literal["NEW", "REFINE"] = "NEW"
    verification_method: Literal[
        "policy_logprob",
        "behavioral_replay",
        "structured_replay",
    ] = "structured_replay"
    verified_success_count: int = 0
    positive_advantage_count: int = 0
    verification_error: str = ""


class ProceduralMemoryState(BaseModel):
    bank_id: str
    skills: dict[str, ProceduralSkill] = Field(default_factory=dict)
    episodes: list[EvaluationEvidence] = Field(default_factory=list)
    experiences: list[SkillExperience] = Field(default_factory=list)
    ppo_decisions: list[PPOGateDecision] = Field(default_factory=list)
    baselines: dict[str, float] = Field(default_factory=dict)
    iteration: int = 0
    evolution_log: list[dict[str, Any]] = Field(default_factory=list)
    maintenance_log: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
