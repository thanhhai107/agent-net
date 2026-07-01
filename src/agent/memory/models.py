"""Skill-Pro inspired procedural skill schemas."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryAttributes(BaseModel):
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    task_stages: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class MemoryQuery(BaseModel):
    text: str
    scenario: str = ""
    topology_class: str = ""
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    task_stage: str = "diagnosis"
    tools: list[str] = Field(default_factory=list)
    top_k: int = 5
    token_budget: int = 1500


class SkillStep(BaseModel):
    order: int
    action: str
    tool_name: str = ""
    arguments_hint: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class SemanticGradient(BaseModel):
    source_session_id: str
    critique: str
    proposed_update: str
    gradient_source: Literal["llm", "deterministic"] = "deterministic"
    created_at: str = Field(default_factory=utc_now)


class EvaluationEvidence(BaseModel):
    session_id: str
    task_description: str = ""
    scenario: str = ""
    topology_class: str = ""
    root_cause: list[str] = Field(default_factory=list)
    faulty_devices: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    steps: int = 0
    tool_calls: int = 0
    success: bool = False


class ProceduralSkill(BaseModel):
    skill_id: str
    title: str
    activation_condition: str
    execution_steps: list[SkillStep] = Field(min_length=1)
    termination_condition: str
    source_sessions: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    status: Literal["candidate", "validated", "retired"] = "candidate"
    reuse_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    score: float = 0.0
    semantic_gradients: list[SemanticGradient] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def memory_id(self) -> str:
        return self.skill_id

    def content_hash(self) -> str:
        payload = self.model_dump(
            exclude={
                "created_at",
                "updated_at",
                "reuse_count",
                "success_count",
                "failure_count",
                "score",
                "source_sessions",
            }
        )
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SkillRetrieval(BaseModel):
    memory: ProceduralSkill
    score: float
    reasons: list[str] = Field(default_factory=list)

    @property
    def skill(self) -> ProceduralSkill:
        return self.memory


class PPOGateDecision(BaseModel):
    accepted: bool
    reason: str
    candidate_score: float
    baseline_score: float
    replaced_skill_id: str | None = None


class SkillMemoryState(BaseModel):
    bank_id: str
    skills: dict[str, ProceduralSkill] = Field(default_factory=dict)
    episodes: list[EvaluationEvidence] = Field(default_factory=list)
    ppo_decisions: list[PPOGateDecision] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
