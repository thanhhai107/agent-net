"""Policy scorers for Skill-Pro trust-region verification.

NIKA tool traces do not expose token log probabilities, so the scorer replays
saved decisions against the frozen LLM and falls back to a deterministic,
structure-first comparison.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from agent.procedural_memory.models import ProceduralSkill, SkillExperience


class PolicyReplayItem(BaseModel):
    experience_id: str
    candidate_alignment: float = Field(ge=0.0, le=1.0)
    baseline_alignment: float = Field(ge=0.0, le=1.0)


class PolicyReplayDraft(BaseModel):
    scores: list[PolicyReplayItem] = Field(default_factory=list)


class PolicyReplayResult(BaseModel):
    scores: list[PolicyReplayItem]
    method: Literal["behavioral_replay", "structured_replay"]
    error: str = ""


class PolicyScorer(Protocol):
    def score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        baseline: ProceduralSkill | None,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult: ...


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", str(value or "").lower())
    }


def _jaccard(left: Any, right: Any) -> float:
    lhs = _tokens(left)
    rhs = _tokens(right)
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def _expected_tools(skill: ProceduralSkill | None) -> list[str]:
    if skill is None:
        return []
    tools = list(skill.tools)
    tools.extend(step.tool_name for step in skill.execution_steps if step.tool_name)
    return list(dict.fromkeys(tool for tool in tools if tool))


def _lcs_ratio(expected: Sequence[str], observed: Sequence[str]) -> float:
    if not expected:
        return 0.5
    if not observed:
        return 0.0
    previous = [0] * (len(observed) + 1)
    for left in expected:
        current = [0]
        for index, right in enumerate(observed, start=1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / max(len(expected), 1)


class StructuredReplayPolicyScorer:
    """Score policy behavior from tool coverage, ordering, and state fit."""

    @staticmethod
    def _score(skill: ProceduralSkill | None, experience: SkillExperience) -> float:
        if skill is None:
            return 0.0
        expected_tools = _expected_tools(skill)
        observed_tools = [
            transition.tool_name
            for transition in experience.transitions
            if transition.tool_name
        ]
        if expected_tools:
            covered = len(set(expected_tools) & set(observed_tools)) / len(
                set(expected_tools)
            )
        else:
            covered = 0.5
        ordering = _lcs_ratio(expected_tools, observed_tools)
        policy_text = " ".join(step.action for step in skill.execution_steps)
        transition_text = " ".join(
            " ".join(
                (
                    transition.action,
                    transition.tool_name,
                    transition.observation_summary,
                )
            )
            for transition in experience.transitions
        )
        semantic_fit = _jaccard(policy_text, transition_text)
        state_fit = _jaccard(
            " ".join(
                (
                    skill.activation_condition,
                    " ".join(skill.protocols),
                    " ".join(skill.services),
                    " ".join(skill.symptoms),
                )
            ),
            " ".join((experience.trajectory, transition_text)),
        )
        score = (
            0.5 * covered
            + 0.25 * ordering
            + 0.15 * semantic_fit
            + 0.10 * state_fit
        )
        return max(0.0, min(1.0, score))

    def score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        baseline: ProceduralSkill | None,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult:
        return PolicyReplayResult(
            scores=[
                PolicyReplayItem(
                    experience_id=experience.experience_id,
                    candidate_alignment=self._score(candidate, experience),
                    baseline_alignment=self._score(baseline, experience),
                )
                for experience in experiences
            ],
            method="structured_replay",
        )


class BehavioralReplayPolicyScorer:
    """Ask the frozen LLM how each Skill changes the recorded action policy."""

    def __init__(
        self,
        llm_factory: Callable[[], Any | None],
        *,
        fallback: PolicyScorer | None = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.fallback = fallback or StructuredReplayPolicyScorer()

    def score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        baseline: ProceduralSkill | None,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult:
        if not experiences:
            return PolicyReplayResult(scores=[], method="behavioral_replay")
        try:
            llm = self.llm_factory()
            if llm is None:
                raise RuntimeError("learning LLM is unavailable")
            payload = [
                {
                    "experience_id": item.experience_id,
                    "state": item.trajectory[:900],
                    "actions": [
                        {
                            "tool": transition.tool_name,
                            "arguments": transition.arguments_hint,
                            "observation": transition.observation_summary[:400],
                            "done": transition.done,
                        }
                        for transition in item.transitions[:10]
                    ],
                }
                for item in experiences
            ]
            prompt = (
                "You are a frozen-policy behavioral replay scorer for Skill-Pro. "
                "For every saved experience, estimate how compatible the recorded "
                "action sequence is with the candidate Skill and with the baseline "
                "Skill at the same visible states. Score policy compatibility from "
                "0 to 1. Do not judge the hidden answer and do not reward word overlap; "
                "focus on activation fit, action/tool ordering, and termination. "
                "Return exactly one score row for every experience_id.\n\n"
                f"Candidate Skill:\n{candidate.format_for_llm()}\n\n"
                f"Baseline Skill:\n{baseline.format_for_llm() if baseline else 'NO SKILL'}\n\n"
                f"Saved experiences:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
            scorer = llm.with_structured_output(PolicyReplayDraft)
            raw = scorer.invoke(prompt)
            draft = (
                raw
                if isinstance(raw, PolicyReplayDraft)
                else PolicyReplayDraft.model_validate(raw)
            )
            expected_ids = {item.experience_id for item in experiences}
            by_id = {
                item.experience_id: item
                for item in draft.scores
                if item.experience_id in expected_ids
            }
            if set(by_id) != expected_ids:
                raise ValueError("behavioral replay omitted or invented experience ids")
            return PolicyReplayResult(
                scores=[by_id[item.experience_id] for item in experiences],
                method="behavioral_replay",
            )
        except Exception as exc:
            fallback = self.fallback.score_batch(
                candidate=candidate,
                baseline=baseline,
                experiences=experiences,
            )
            fallback.error = f"{type(exc).__name__}: {exc}"
            return fallback
