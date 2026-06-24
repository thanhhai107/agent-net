"""Structured models shared by the advanced LangGraph workflows."""

from pydantic import BaseModel, Field, model_validator


class PlanStep(BaseModel):
    """One concrete investigation step."""

    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)


class InvestigationPlan(BaseModel):
    """Initial structured troubleshooting plan."""

    objective: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)


class StepResult(BaseModel):
    """Evidence returned by the executor for one plan step."""

    step: PlanStep
    observation: str
    succeeded: bool = True


class ReplanDecision(BaseModel):
    """Decision to finish diagnosis or continue with a revised plan."""

    completed: bool
    diagnosis_report: str = ""
    remaining_steps: list[PlanStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "ReplanDecision":
        if self.completed and not self.diagnosis_report.strip():
            raise ValueError("diagnosis_report is required when completed is true")
        if not self.completed and not self.remaining_steps:
            raise ValueError("remaining_steps are required when completed is false")
        return self


class ReflexionEvaluation(BaseModel):
    """Evaluator verdict for one complete troubleshooting attempt."""

    success: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    evidence_sufficient: bool
    anomaly_assessment: str = Field(min_length=1)
    localization_assessment: str = Field(min_length=1)
    root_cause_assessment: str = Field(min_length=1)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_success(self) -> "ReflexionEvaluation":
        if self.success and not self.evidence_sufficient:
            raise ValueError("a successful evaluation requires sufficient evidence")
        if self.success and self.quality_score < 0.8:
            raise ValueError("a successful evaluation requires quality_score >= 0.8")
        if not self.success and not (
            self.contradictions or self.missing_evidence or self.failure_reasons
        ):
            raise ValueError("an unsuccessful evaluation requires actionable feedback")
        return self


class ReflexionMemory(BaseModel):
    """Compact episodic memory generated after an unsuccessful attempt."""

    summary: str = Field(min_length=1)
    lessons: list[str] = Field(min_length=1)
    next_strategy: list[str] = Field(min_length=1)
    evidence_to_collect: list[str] = Field(default_factory=list)
    avoid_repeating: list[str] = Field(default_factory=list)
