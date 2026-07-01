"""Pydantic schemas for DRAFT-style tool documentation refinement."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


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


class DocumentationRevision(BaseModel):
    revision_id: str
    tool_name: str
    before_hash: str
    after_hash: str
    changed: bool
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class DraftRewriteProposal(BaseModel):
    """LLM-produced DRAFT rewrite for one primitive tool's documentation."""

    tool_name: str
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    parameters: dict[str, ToolParameterDoc] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    usage_notes: list[str] = Field(default_factory=list)
    positive_examples: list[dict[str, Any]] = Field(default_factory=list)
    negative_examples: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""


class ToolDocumentation(BaseModel):
    name: str
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    parameters: dict[str, ToolParameterDoc] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    usage_notes: list[str] = Field(default_factory=list)
    positive_examples: list[dict[str, Any]] = Field(default_factory=list)
    negative_examples: list[dict[str, Any]] = Field(default_factory=list)
    version: int = 1
    frozen: bool = False
    frozen_reason: str = ""
    updated_at: str = Field(default_factory=utc_now)

    def content_hash(self) -> str:
        payload = self.model_dump(
            exclude={"updated_at", "version", "frozen", "frozen_reason"}
        )
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def refined_description(self, *, max_chars: int = 1600) -> str:
        parts = [self.description.strip() or f"Primitive diagnostic tool `{self.name}`."]
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


class DraftToolState(BaseModel):
    library_id: str
    documents: dict[str, ToolDocumentation] = Field(default_factory=dict)
    trials: list[ToolTrial] = Field(default_factory=list)
    gaps: list[ComprehensionGap] = Field(default_factory=list)
    revisions: list[DocumentationRevision] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def tool_trials(self, tool_name: str) -> list[ToolTrial]:
        return [trial for trial in self.trials if trial.tool_name == tool_name]
