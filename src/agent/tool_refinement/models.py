"""Pydantic schemas for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolParameterDoc(BaseModel):
    name: str
    type_hint: str = "unknown"
    description: str = ""
    constraints: list[str] = Field(default_factory=list)
    examples: list[Any] = Field(default_factory=list)


class ToolTrial(BaseModel):
    trial_id: str
    session_id: str
    tool_name: str
    task_description: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error", "unknown"] = "unknown"
    output_summary: str = ""
    error_summary: str = ""
    timestamp: str = Field(default_factory=utc_now)

    @property
    def success(self) -> bool:
        return self.status == "success"


class ComprehensionGap(BaseModel):
    gap_id: str
    tool_name: str
    gap_type: str
    evidence: str
    recommendation: str
    session_id: str = ""
    created_at: str = Field(default_factory=utc_now)


class DraftExploration(BaseModel):
    exploration_id: str
    session_id: str
    trial_id: str = ""
    tool_name: str
    intent: Literal[
        "unknown",
        "tool_validation",
        "boundary_case",
        "argument_schema_probe",
    ] = "unknown"
    user_query: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    status: Literal[
        "success",
        "error",
        "unknown",
        "invalidated",
    ] = "unknown"
    document_hash: str = ""
    analyzer_suggestion: str = ""
    diversity_score: float = 1.0
    reflection_count: int = 0
    read_only: bool = True
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_diagnosis_plan(_cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        migrated.pop("scope", None)
        if migrated.get("intent") == "diagnosis_check":
            migrated["intent"] = "tool_validation"
        if migrated.get("status") == "planned":
            migrated["status"] = "invalidated"
        elif migrated.get("status") == "consumed":
            migrated["status"] = "unknown"
        return migrated


class DraftAnalyzerSuggestion(BaseModel):
    suggestion_id: str
    tool_name: str
    session_id: str = ""
    trial_ids: list[str] = Field(default_factory=list)
    suggestion: str
    created_at: str = Field(default_factory=utc_now)


class DraftAnalyzerDraft(BaseModel):
    """Structured natural-language analysis of one tool's exploration batch."""

    suggestion: str = ""
    rationale: str = ""


class DraftExplorerDraft(BaseModel):
    """One self-driven, single-call DRAFT exploration proposal."""

    user_query: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    intent: Literal[
        "tool_validation",
        "boundary_case",
        "argument_schema_probe",
    ] = "tool_validation"
    rationale: str = ""


class DocumentationRevision(BaseModel):
    revision_id: str
    tool_name: str
    source_signature: str = ""
    before_hash: str
    after_hash: str
    changed: bool
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)
    analyzer_suggestion_ids: list[str] = Field(default_factory=list)
    llm_error: str = ""
    created_at: str = Field(default_factory=utc_now)


class DraftRewriteProposal(BaseModel):
    """LLM-produced DRAFT rewrite for one primitive tool's documentation."""

    tool_name: str
    description: str = ""
    tool_usage_description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    parameters: dict[str, ToolParameterDoc] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    usage_notes: list[str] = Field(default_factory=list)
    positive_examples: list[dict[str, Any]] = Field(default_factory=list)
    negative_examples: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    next_exploration_direction: str = ""


class DraftRewriteDraft(BaseModel):
    """Small DRAFT rewrite payload used for bounded training LLM calls."""

    tool_name: str = ""
    description: str = ""
    tool_usage_description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    parameters: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    usage_notes: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    next_exploration_direction: str = ""


class ToolDocumentation(BaseModel):
    name: str
    description: str = ""
    source_signature: str = ""
    source_schema: dict[str, Any] = Field(default_factory=dict)
    source_contract_version: int = 0
    tool_usage_description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    parameters: dict[str, ToolParameterDoc] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    usage_notes: list[str] = Field(default_factory=list)
    positive_examples: list[dict[str, Any]] = Field(default_factory=list)
    negative_examples: list[dict[str, Any]] = Field(default_factory=list)
    rewrite_history: list[str] = Field(default_factory=list)
    analyzer_suggestions: list[str] = Field(default_factory=list)
    next_exploration_direction: str = ""
    trial_count: int = 0
    success_count: int = 0
    error_count: int = 0
    mastery_score: float = 0.0
    contract_mastery_score: float = 0.0
    diagnostic_utility_score: float = 0.0
    diagnostic_utility_count: int = 0
    diagnostic_utility_sessions: list[str] = Field(default_factory=list)
    last_convergence_score: float = 0.0
    version: int = 1
    published: bool = False
    frozen: bool = False
    frozen_reason: str = ""
    updated_at: str = Field(default_factory=utc_now)

    def content_hash(self) -> str:
        payload = self.model_dump(
            exclude={
                "updated_at",
                "version",
                "published",
                "frozen",
                "frozen_reason",
                "rewrite_history",
                "analyzer_suggestions",
                "trial_count",
                "success_count",
                "error_count",
                "mastery_score",
                "contract_mastery_score",
                "diagnostic_utility_score",
                "diagnostic_utility_count",
                "diagnostic_utility_sessions",
                "last_convergence_score",
                "source_signature",
            }
        )
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def refined_description(self, *, max_chars: int = 1600) -> str:
        summary = (
            self.tool_usage_description.strip()
            or self.description.strip()
            or f"Primitive diagnostic tool `{self.name}`."
        )
        parts = [summary]
        if self.tool_usage_description and self.description:
            parts.append("Functionality: " + self.description.strip())
        if self.preconditions:
            parts.append("Preconditions: " + "; ".join(self.preconditions[:4]))
        if self.parameters:
            params = []
            for param in self.parameters.values():
                text = param.name
                if param.type_hint != "unknown":
                    text += f" ({param.type_hint})"
                if param.description:
                    text += f": {param.description}"
                if param.constraints:
                    text += " [" + "; ".join(param.constraints[:3]) + "]"
                params.append(text)
            parts.append("Parameters: " + " | ".join(params[:8]))
        if self.constraints:
            parts.append("Constraints: " + "; ".join(self.constraints[:5]))
        if self.failure_modes:
            parts.append("Known failure modes: " + "; ".join(self.failure_modes[:4]))
        if self.usage_notes:
            parts.append("Usage notes: " + "; ".join(self.usage_notes[:5]))
        text = "\n".join(part for part in parts if part).strip()
        return text[:max_chars]


class DraftToolStats(BaseModel):
    tool_name: str
    trials: int = 0
    successes: int = 0
    errors: int = 0
    gaps: int = 0
    revisions: int = 0
    llm_rewrites: int = 0
    explorations: int = 0
    mastery_score: float = 0.0
    contract_mastery_score: float = 0.0
    diagnostic_utility_score: float = 0.0
    convergence_score: float = 0.0
    documented_path_rate: float = 0.0
    success_path_rate: float = 0.0
    mastered: bool = False
    updated_at: str = Field(default_factory=utc_now)


class DraftToolState(BaseModel):
    library_id: str
    documents: dict[str, ToolDocumentation] = Field(default_factory=dict)
    trials: list[ToolTrial] = Field(default_factory=list)
    explorations: list[DraftExploration] = Field(default_factory=list)
    analyzer_suggestions: list[DraftAnalyzerSuggestion] = Field(default_factory=list)
    gaps: list[ComprehensionGap] = Field(default_factory=list)
    revisions: list[DocumentationRevision] = Field(default_factory=list)
    processed_trial_ids: list[str] = Field(default_factory=list)
    tool_stats: dict[str, DraftToolStats] = Field(default_factory=dict)
    library_usage_description: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def tool_trials(self, tool_name: str) -> list[ToolTrial]:
        return [trial for trial in self.trials if trial.tool_name == tool_name]
