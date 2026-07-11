"""Evidence-bound models and validation for final benchmark submissions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from agent.langgraph.evidence import ToolObservation


class SupportedClaim(BaseModel):
    """One proposed value and the current-run observations cited for it."""

    value: str = Field(description="Exact device name or root-cause id.")
    evidence_ids: list[str] = Field(default_factory=list)


class DiagnosisDraft(BaseModel):
    """A model-produced submission draft before evidence verification."""

    anomaly_status: Literal["present", "absent", "inconclusive"]
    anomaly_evidence_ids: list[str] = Field(default_factory=list)
    faulty_devices: list[SupportedClaim] = Field(default_factory=list)
    root_causes: list[SupportedClaim] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim_id: str
    supported: bool
    reason: str = ""


class SubmissionVerification(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    tool: str
    tool_input: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ValidatedSubmission:
    is_anomaly: bool
    faulty_devices: tuple[str, ...]
    root_cause_name: tuple[str, ...]
    unsupported_claims: tuple[str, ...] = ()

    def to_tool_args(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "faulty_devices": list(self.faulty_devices),
            "root_cause_name": list(self.root_cause_name),
        }


def evidence_records(
    observations: Sequence[ToolObservation],
    *,
    max_records: int = 40,
    max_summary_chars: int = 2_000,
) -> list[EvidenceRecord]:
    """Normalize and deduplicate current-run observations with stable local ids."""
    records: list[EvidenceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for observation in observations:
        tool = str(observation.tool or "").strip()
        tool_input = str(observation.tool_input or "").strip()
        summary = str(observation.summary or "").strip()
        if not tool or not summary:
            continue
        key = (tool, tool_input, summary)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            EvidenceRecord(
                evidence_id=f"ev-{len(records) + 1:03d}",
                tool=tool,
                tool_input=tool_input[:2_000],
                summary=summary[:max_summary_chars],
            )
        )
        if len(records) >= max_records:
            break
    return records


def draft_claims(
    draft: DiagnosisDraft,
    *,
    available_root_causes: Sequence[str],
    valid_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    """Convert a draft into verifier claims after structural validation."""
    claims: list[dict[str, Any]] = []
    anomaly_ids = _valid_citations(draft.anomaly_evidence_ids, valid_evidence_ids)
    if draft.anomaly_status != "inconclusive" and anomaly_ids:
        claims.append(
            {
                "claim_id": "anomaly",
                "kind": "anomaly",
                "value": draft.anomaly_status,
                "evidence_ids": anomaly_ids,
            }
        )

    if draft.anomaly_status != "present":
        return claims

    claims.extend(
        _value_claims(
            kind="device",
            proposed=draft.faulty_devices,
            valid_evidence_ids=valid_evidence_ids,
        )
    )
    claims.extend(
        _value_claims(
            kind="root_cause",
            proposed=draft.root_causes,
            valid_evidence_ids=valid_evidence_ids,
            allowed_values=set(available_root_causes),
        )
    )
    return claims


def validated_submission(
    draft: DiagnosisDraft,
    claims: Sequence[dict[str, Any]],
    verification: SubmissionVerification,
) -> ValidatedSubmission:
    """Keep supported claims and fall back to a conservative healthy verdict.

    A completed diagnosis must still produce an evaluable submission. Unverified
    anomaly evidence therefore means ``is_anomaly=False`` rather than no
    submission; execution failures and recursion exhaustion are handled before
    this function and remain non-submissions.
    """
    verdicts = {
        verdict.claim_id: verdict.supported for verdict in verification.verdicts
    }
    supported = {
        str(claim["claim_id"]): claim
        for claim in claims
        if verdicts.get(str(claim["claim_id"]), False)
    }
    unsupported = tuple(
        str(claim["claim_id"])
        for claim in claims
        if str(claim["claim_id"]) not in supported
    )
    anomaly = supported.get("anomaly")
    if anomaly is None:
        return ValidatedSubmission(False, (), (), unsupported)
    if anomaly["value"] == "absent":
        return ValidatedSubmission(False, (), (), unsupported)
    if anomaly["value"] != "present":
        return ValidatedSubmission(False, (), (), unsupported)

    devices = _supported_values(supported, "device")
    root_causes = _supported_values(supported, "root_cause")
    return ValidatedSubmission(True, devices, root_causes, unsupported)


def submission_draft_prompt(
    *,
    task_description: str,
    diagnosis_report: str,
    records: Sequence[EvidenceRecord],
    available_root_causes: Sequence[str],
) -> str:
    payload = {
        "task": task_description[:20_000],
        "diagnosis_report": diagnosis_report[:60_000],
        "available_root_causes": list(available_root_causes),
        "current_run_evidence": [record.to_dict() for record in records],
    }
    return (
        "Create a minimal diagnosis draft from the payload. Every claim must cite "
        "one or more evidence_id values whose observations directly support it. "
        "Task wording, the report itself, prior memory, skills, and tool docs are "
        "not evidence. Use anomaly_status='present' only for observed abnormal "
        "behavior, 'absent' only for concrete observations establishing normal "
        "behavior, otherwise 'inconclusive'. Device values must be exact observed "
        "device names. Root causes must exactly match available_root_causes. Do not "
        "include alternatives or weak hypotheses.\n\n"
        + json.dumps(payload, ensure_ascii=True, default=str)
    )


def verification_prompt(
    *,
    records: Sequence[EvidenceRecord],
    claims: Sequence[dict[str, Any]],
) -> str:
    evidence_by_id = {record.evidence_id: record.to_dict() for record in records}
    cited_ids = {
        evidence_id
        for claim in claims
        for evidence_id in claim.get("evidence_ids", [])
    }
    payload = {
        "claims": list(claims),
        "cited_evidence": [
            evidence_by_id[evidence_id]
            for evidence_id in sorted(cited_ids)
            if evidence_id in evidence_by_id
        ],
    }
    return (
        "Independently verify each claim using only its cited tool observations. "
        "Return exactly one verdict for every claim_id. Mark supported=false when "
        "the evidence is indirect, ambiguous, contradictory, merely shows a "
        "symptom that does not identify the claimed device/cause, or does not "
        "establish healthy behavior for anomaly=absent. Do not infer from task "
        "wording or networking plausibility.\n\n"
        + json.dumps(payload, ensure_ascii=True, default=str)
    )


def _valid_citations(values: Sequence[str], valid_ids: set[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value in valid_ids))


def _value_claims(
    *,
    kind: str,
    proposed: Sequence[SupportedClaim],
    valid_evidence_ids: set[str],
    allowed_values: set[str] | None = None,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in proposed:
        value = item.value.strip()
        citations = _valid_citations(item.evidence_ids, valid_evidence_ids)
        if (
            not value
            or value in seen
            or not citations
            or (allowed_values is not None and value not in allowed_values)
        ):
            continue
        seen.add(value)
        claims.append(
            {
                "claim_id": f"{kind}:{len(claims) + 1}",
                "kind": kind,
                "value": value,
                "evidence_ids": citations,
            }
        )
    return claims


def _supported_values(
    supported: dict[str, dict[str, Any]], kind: str
) -> tuple[str, ...]:
    values: list[str] = []
    for claim in supported.values():
        value = str(claim.get("value") or "").strip()
        if claim.get("kind") == kind and value and value not in values:
            values.append(value)
    return tuple(values)
