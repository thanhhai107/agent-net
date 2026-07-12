"""Structured decisions used by local diagnostic workflows."""

from pydantic import BaseModel, Field, model_validator


class PlanStep(BaseModel):
    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)


class InvestigationPlan(BaseModel):
    objective: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)


class ReplanDecision(BaseModel):
    completed: bool
    diagnosis_report: str = ""
    remaining_steps: list[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "ReplanDecision":
        if self.completed and not self.diagnosis_report.strip():
            raise ValueError("a completed plan requires a diagnosis report")
        if not self.completed and not self.remaining_steps:
            raise ValueError("an incomplete plan requires remaining steps")
        return self


class ReflexionEvaluation(BaseModel):
    success: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    evidence_sufficient: bool
    feedback: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_success(self) -> "ReflexionEvaluation":
        if self.success and (not self.evidence_sufficient or self.quality_score < 0.8):
            raise ValueError("success requires sufficient evidence and score >= 0.8")
        if not self.success and not self.feedback:
            raise ValueError("an unsuccessful attempt requires actionable feedback")
        return self


class ReflexionLesson(BaseModel):
    lessons: list[str] = Field(min_length=1)
    next_strategy: list[str] = Field(min_length=1)
