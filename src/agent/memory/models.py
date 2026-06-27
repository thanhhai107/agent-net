"""Models for atomic procedural memory.

The module is intentionally organized around three ideas:
- MemInsight-style context attributes for retrieval.
- LightMem-style post-episode validation/consolidation.
- A-Mem-style atomic notes and graph links.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MemoryStatus(StrEnum):
    STAGED = "staged"
    VALIDATED = "validated"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class MemoryLinkType(StrEnum):
    SUPPORTS = "supports"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    SAME_PATTERN = "same_pattern"


class MemoryAttributes(BaseModel):
    """MemInsight-style searchable, non-oracle attributes."""

    scenarios: list[str] = Field(default_factory=list)
    topology_classes: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    task_stages: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    def normalized(self) -> "MemoryAttributes":
        values: dict[str, list[str]] = {}
        for field_name in type(self).model_fields:
            raw_values = getattr(self, field_name)
            values[field_name] = sorted(
                {
                    str(value).strip().lower()
                    for value in raw_values
                    if str(value).strip()
                }
            )
        return MemoryAttributes(**values)

    def flat_values(self) -> set[str]:
        normalized = self.normalized()
        return {
            value
            for field_name in type(normalized).model_fields
            for value in getattr(normalized, field_name)
        }


class MemoryCandidate(BaseModel):
    """One A-Mem-style atomic procedural note proposed by the extractor."""

    content: str = Field(min_length=12, max_length=1200)
    applicability: list[str] = Field(default_factory=list, max_length=8)
    evidence_required: list[str] = Field(default_factory=list, max_length=8)
    avoid: list[str] = Field(default_factory=list, max_length=8)
    attributes: MemoryAttributes = Field(default_factory=MemoryAttributes)

    @model_validator(mode="after")
    def ensure_procedural_value(self) -> "MemoryCandidate":
        if not (self.applicability or self.evidence_required or self.avoid):
            raise ValueError(
                "a procedural memory requires applicability, evidence, or an avoid rule"
            )
        self.attributes = self.attributes.normalized()
        return self


class MemoryExtraction(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list, max_length=6)


class EvaluationEvidence(BaseModel):
    """Only numeric benchmark feedback exposed to the validation policy."""

    detection_score: float
    localization_f1: float
    rca_f1: float

    @property
    def fully_successful(self) -> bool:
        return (
            self.detection_score >= 1.0
            and self.localization_f1 >= 1.0
            and self.rca_f1 >= 1.0
        )

    @property
    def aggregate_score(self) -> float:
        scores = [
            max(0.0, min(1.0, self.detection_score)),
            max(0.0, min(1.0, self.localization_f1)),
            max(0.0, min(1.0, self.rca_f1)),
        ]
        return sum(scores) / len(scores)


class MemoryQuery(BaseModel):
    text: str = Field(min_length=1)
    scenario: str = ""
    topology_class: str = ""
    protocols: list[str] = Field(default_factory=list)
    task_stage: str = "diagnosis"
    tools: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_limit: int = Field(default=20, ge=1, le=100)
    token_budget: int = Field(default=1500, ge=100, le=12000)

    def attributes(self) -> MemoryAttributes:
        return MemoryAttributes(
            scenarios=[self.scenario] if self.scenario else [],
            topology_classes=[self.topology_class] if self.topology_class else [],
            protocols=self.protocols,
            task_stages=[self.task_stage] if self.task_stage else [],
            tools=self.tools,
        ).normalized()


class StoredMemory(MemoryCandidate):
    memory_id: str
    bank_id: str
    status: MemoryStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source_session_id: str
    version: int = Field(ge=1)
    validation_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    created_at: str
    superseded_at: str | None = None
    superseded_by: str | None = None

    def embedding_text(self) -> str:
        attrs = self.attributes.normalized()
        sections = [
            f"Atomic procedural note: {self.content}",
        ]
        if self.applicability:
            sections.append("Applies when: " + "; ".join(self.applicability))
        if self.evidence_required:
            sections.append("Evidence required: " + "; ".join(self.evidence_required))
        if self.avoid:
            sections.append("Avoid: " + "; ".join(self.avoid))
        for label, values in (
            ("Protocols", attrs.protocols),
            ("Services", attrs.services),
            ("Task stages", attrs.task_stages),
            ("Symptoms", attrs.symptoms),
            ("Tools", attrs.tools),
        ):
            if values:
                sections.append(f"{label}: " + ", ".join(values))
        return "\n".join(sections)


class RetrievedMemory(BaseModel):
    memory: StoredMemory
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    attribute_score: float = 0.0
    graph_score: float = 0.0


class MemoryRelationDecision(BaseModel):
    relation: MemoryLinkType | None = None
    reason: str = Field(default="", max_length=300)


class MemorySnapshot(BaseModel):
    bank_id: str
    session_id: str
    created_at: str
    memories: list[dict[str, Any]]
    links: list[dict[str, Any]]
