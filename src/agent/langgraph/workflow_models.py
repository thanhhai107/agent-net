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


class DiagnosisCritique(BaseModel):
    """Structured quality review of a diagnosis report."""

    evidence_sufficient: bool
    anomaly_assessment: str = Field(min_length=1)
    localization_assessment: str = Field(min_length=1)
    root_cause_assessment: str = Field(min_length=1)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(min_length=1)
