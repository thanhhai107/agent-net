"""Data models for mastered primitives, generated tools, and composites."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolEvolutionMode(StrEnum):
    MASTERY = "mastery"
    DISTILL = "distill"
    DUAL = "dual"

    @property
    def mastery_enabled(self) -> bool:
        return self in {self.MASTERY, self.DUAL}

    @property
    def distillation_enabled(self) -> bool:
        return self in {self.DISTILL, self.DUAL}

    @property
    def validation_enabled(self) -> bool:
        return True

    @property
    def dedup_enabled(self) -> bool:
        return True


class ToolParameter(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    type: Literal["str", "int", "float", "bool"] = "str"
    description: str = Field(min_length=3, max_length=300)
    required: bool = True
    default: Any | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_json_schema_type(cls, value: Any) -> Any:
        aliases = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
        }
        return aliases.get(str(value).lower(), value)


class CompositeStep(BaseModel):
    tool: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    label: str = Field(default="", max_length=160)


class ValidationEvidence(BaseModel):
    context_fingerprint: str
    execution_success: bool
    incident_success: bool
    source: Literal[
        "runtime",
        "synthesis",
        "distillation",
        "generation",
        "replay",
    ] = "runtime"
    structural_valid: bool = True
    semantic_valid: bool = True
    observed_at: str = Field(default_factory=utc_now)


class ToolVerificationReport(BaseModel):
    stage: Literal["structural", "runtime", "sandbox", "semantic", "replay"]
    passed: bool
    checks: list[str] = Field(default_factory=list)
    error: str | None = None
    context_fingerprint: str = ""
    observed_at: str = Field(default_factory=utc_now)


class CapabilityGap(BaseModel):
    gap_id: str
    description: str = Field(min_length=12, max_length=600)
    required_inputs: list[str] = Field(default_factory=list, max_length=12)
    expected_observations: list[str] = Field(default_factory=list, max_length=12)
    status: Literal["open", "synthesized", "resolved", "abandoned"] = "open"
    proposed_tool: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class CompositeTool(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    description: str = Field(min_length=12, max_length=1000)
    parameters: list[ToolParameter] = Field(default_factory=list, max_length=16)
    steps: list[CompositeStep] = Field(min_length=1, max_length=8)
    output_contract: list[str] = Field(default_factory=list, max_length=12)
    tags: list[str] = Field(default_factory=list, max_length=12)
    status: Literal["ephemeral", "draft", "candidate", "promoted", "rejected"] = "draft"
    version: int = Field(default=1, ge=1)
    parent_name: str | None = None
    signature: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    evidence: list[ValidationEvidence] = Field(default_factory=list)
    verification_reports: list[ToolVerificationReport] = Field(default_factory=list)
    execution_count: int = 0
    success_count: int = 0
    retrieval_count: int = 0
    source_trace_hash: str = ""
    last_used_at: str | None = None

    @field_validator("parameters")
    @classmethod
    def unique_parameters(cls, value: list[ToolParameter]) -> list[ToolParameter]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Composite tool parameter names must be unique.")
        return value

    def canonical_signature(self) -> str:
        aliases = {item.name: f"p{index}" for index, item in enumerate(self.parameters)}

        def normalize(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, str):
                for name, alias in aliases.items():
                    value = value.replace(f"${{{name}}}", f"${{{alias}}}")
            return value

        payload = {
            "parameter_types": [item.type for item in self.parameters],
            "steps": [
                {
                    "tool": step.tool,
                    "arguments": normalize(step.arguments),
                }
                for step in self.steps
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def ensure_signature(self) -> "CompositeTool":
        if not self.signature:
            self.signature = self.canonical_signature()
        return self

    def utility_score(self) -> float:
        if self.status == "rejected":
            return -100.0
        score = (
            self.success_count * 2.0
            - max(self.execution_count - self.success_count, 0) * 3.0
        )
        score += min(self.retrieval_count, 20) * 0.1
        if self.status == "promoted":
            score += 5.0
        if any(
            report.passed and report.stage == "replay"
            for report in self.verification_reports
        ):
            score += 2.0
        return score


class GeneratedTool(BaseModel):
    """TTE/Alita-style generated executable tool stored as Python source."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    code: str = Field(min_length=12, max_length=20000)
    description: str = Field(min_length=12, max_length=1000)
    parameters: list[ToolParameter] = Field(default_factory=list, max_length=16)
    output_description: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=12)
    status: Literal["ephemeral", "candidate", "promoted", "rejected"] = "ephemeral"
    version: int = Field(default=1, ge=1)
    parent_name: str | None = None
    signature: str = ""
    dependencies: list[str] = Field(default_factory=list, max_length=12)
    test_example: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    evidence: list[ValidationEvidence] = Field(default_factory=list)
    verification_reports: list[ToolVerificationReport] = Field(default_factory=list)
    execution_count: int = 0
    success_count: int = 0
    retrieval_count: int = 0
    source_trace_hash: str = ""
    last_used_at: str | None = None

    @field_validator("parameters")
    @classmethod
    def unique_parameters(cls, value: list[ToolParameter]) -> list[ToolParameter]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Generated tool parameter names must be unique.")
        return value

    def canonical_signature(self) -> str:
        normalized_code = " ".join(self.code.split())
        payload = {
            "code": normalized_code,
            "parameter_types": [item.type for item in self.parameters],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def ensure_signature(self) -> "GeneratedTool":
        if not self.signature:
            self.signature = self.canonical_signature()
        return self

    def utility_score(self) -> float:
        if self.status == "rejected":
            return -100.0
        score = (
            self.success_count * 2.0
            - max(self.execution_count - self.success_count, 0) * 3.0
        )
        score += min(self.retrieval_count, 20) * 0.1
        if self.status == "promoted":
            score += 5.0
        return score


class ToolUsageExample(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool


class ToolCardRevision(BaseModel):
    version: int = Field(ge=1)
    source: Literal["runtime", "explorer", "analyzer", "rewriter"] = "rewriter"
    rationale: str = Field(default="", max_length=500)
    evidence_hash: str = ""
    utility_delta: float = 0.0
    accepted: bool = True
    created_at: str = Field(default_factory=utc_now)


class ToolMastery(BaseModel):
    tool_name: str
    version: int = Field(default=1, ge=1)
    preconditions: list[str] = Field(default_factory=list)
    parameter_guidance: list[str] = Field(default_factory=list)
    output_interpretation: list[str] = Field(default_factory=list)
    failure_semantics: list[str] = Field(default_factory=list)
    usage_examples: list[ToolUsageExample] = Field(default_factory=list)
    calls: int = 0
    successes: int = 0
    errors: int = 0
    source_models: list[str] = Field(default_factory=list)
    revisions: list[ToolCardRevision] = Field(default_factory=list)
    convergence_count: int = 0
    updated_at: str = Field(default_factory=utc_now)

    def agent_overlay(self) -> str:
        sections: list[str] = []
        if self.preconditions:
            sections.append(
                "Observed preconditions: " + "; ".join(self.preconditions[-4:])
            )
        if self.parameter_guidance:
            sections.append(
                "Parameter guidance: " + "; ".join(self.parameter_guidance[-4:])
            )
        if self.output_interpretation:
            sections.append(
                "Output interpretation: " + "; ".join(self.output_interpretation[-4:])
            )
        if self.failure_semantics:
            sections.append("Known failures: " + "; ".join(self.failure_semantics[-4:]))
        if self.usage_examples:
            examples = [
                {
                    "arguments": item.arguments,
                    "succeeded": item.succeeded,
                }
                for item in self.usage_examples[-2:]
            ]
            sections.append(
                "Sanitized examples: "
                + json.dumps(examples, ensure_ascii=False, default=str)
            )
        if self.calls:
            sections.append(
                f"Experience: {self.successes}/{self.calls} observed calls succeeded; "
                f"{self.errors} errors."
            )
        return "\n".join(sections)


class ToolLibraryState(BaseModel):
    schema_version: int = 3
    library_id: str
    mastery: dict[str, ToolMastery] = Field(default_factory=dict)
    composites: dict[str, CompositeTool] = Field(default_factory=dict)
    generated_tools: dict[str, GeneratedTool] = Field(default_factory=dict)
    capability_gaps: dict[str, CapabilityGap] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now)
