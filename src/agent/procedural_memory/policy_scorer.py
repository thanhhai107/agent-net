"""Policy scorers for Skill-Pro trust-region verification.

When the model API exposes echoed prompt log-probabilities, historical actions
are teacher-forced against candidate and baseline skills. Other providers use
behavioral or deterministic replay and are reported as such.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import requests
from pydantic import BaseModel, Field

from agent.procedural_memory.models import ProceduralSkill, SkillExperience
from agent.procedural_memory.policy_context import (
    build_runtime_skill_policy_prefix,
    build_skill_policy_prefix,
)


class PolicyReplayItem(BaseModel):
    experience_id: str
    candidate_alignment: float = Field(ge=0.0, le=1.0)
    baseline_alignment: float = Field(ge=0.0, le=1.0)


class PolicyReplayDraft(BaseModel):
    scores: list[PolicyReplayItem] = Field(default_factory=list)


class PolicyStepLogprob(BaseModel):
    experience_id: str
    transition_index: int
    candidate_logprob: float
    baseline_logprob: float


class PolicyReplayResult(BaseModel):
    scores: list[PolicyReplayItem]
    method: Literal["policy_logprob", "behavioral_replay", "structured_replay"]
    error: str = ""
    step_logprobs: list[PolicyStepLogprob] = Field(default_factory=list)


class PolicyScorer(Protocol):
    def score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        baseline: ProceduralSkill | None,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult: ...


class PolicyLogprobScorer:
    """Teacher-force historical actions through an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _prefix(
        state: str,
        skill: ProceduralSkill | None,
        *,
        policy_token_budget: int = 0,
    ) -> str:
        if policy_token_budget > 0:
            return build_runtime_skill_policy_prefix(
                skill,
                max_tokens=policy_token_budget,
            )
        return build_skill_policy_prefix(state, skill)

    def _score_targets(
        self,
        rows: Sequence[tuple[str, str]],
    ) -> list[float]:
        if not rows:
            return []
        prompts = [prefix + target for prefix, target in rows]
        response = requests.post(
            self.base_url + "/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "prompt": prompts,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or len(choices) != len(rows):
            raise ValueError("logprob API returned an incomplete prompt batch")
        by_index = {int(choice.get("index", -1)): choice for choice in choices}
        scores: list[float] = []
        for index, (prefix, _) in enumerate(rows):
            choice = by_index.get(index)
            if not isinstance(choice, dict):
                raise ValueError(f"logprob API omitted prompt index {index}")
            logprobs = choice.get("logprobs")
            offsets = (
                logprobs.get("text_offset") if isinstance(logprobs, dict) else None
            )
            token_logprobs = (
                logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
            )
            if not isinstance(offsets, list) or not isinstance(token_logprobs, list):
                raise ValueError("logprob API omitted echoed token scores")
            target_scores = [
                float(logprob)
                for offset, logprob in zip(offsets, token_logprobs, strict=True)
                if int(offset) >= len(prefix) and logprob is not None
            ]
            if not target_scores or not all(
                math.isfinite(item) for item in target_scores
            ):
                raise ValueError(
                    f"logprob API could not align target at prompt index {index}"
                )
            # Match Skill-Pro's target-token normalization. Summing here makes
            # PPO ratios depend on serialized action length and over-clips long
            # tool calls even when their mean token likelihood is unchanged.
            scores.append(sum(target_scores) / len(target_scores))
        return scores

    def score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        baseline: ProceduralSkill | None,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult:
        try:
            return self._score_batch(
                candidate=candidate,
                experiences=experiences,
            )
        except Exception as exc:
            return PolicyReplayResult(
                scores=[],
                method="policy_logprob",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _score_batch(
        self,
        *,
        candidate: ProceduralSkill,
        experiences: Sequence[SkillExperience],
    ) -> PolicyReplayResult:
        indexed = [
            (experience, index, transition)
            for experience in experiences
            for index, transition in enumerate(experience.transitions)
        ]
        for _, _, transition in indexed:
            if not transition.policy_context:
                raise ValueError(
                    "historical transition has no pre-action policy context"
                )
        candidate_rows = [
            (
                self._prefix(
                    transition.state,
                    candidate,
                    policy_token_budget=transition.policy_token_budget,
                ),
                transition.action,
            )
            for _, _, transition in indexed
        ]
        baseline_rows = [
            (transition.policy_context, transition.action)
            for _, _, transition in indexed
        ]
        candidate_scores = self._score_targets(candidate_rows)
        baseline_scores = self._score_targets(baseline_rows)
        step_scores = [
            PolicyStepLogprob(
                experience_id=experience.experience_id,
                transition_index=index,
                candidate_logprob=candidate_score,
                baseline_logprob=baseline_score,
            )
            for (experience, index, _), candidate_score, baseline_score in zip(
                indexed,
                candidate_scores,
                baseline_scores,
                strict=True,
            )
        ]
        return PolicyReplayResult(
            scores=[],
            method="policy_logprob",
            step_logprobs=step_scores,
        )


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z0-9_]{3,}", str(value or "").lower())
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
                    transition.state,
                    transition.action,
                    transition.tool_name,
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
        score = 0.5 * covered + 0.25 * ordering + 0.15 * semantic_fit + 0.10 * state_fit
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
    ) -> None:
        self.llm_factory = llm_factory

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
                raise RuntimeError("training LLM is unavailable")
            payload = [
                {
                    "experience_id": item.experience_id,
                    "state": item.trajectory[:900],
                    "actions": [
                        {
                            "state": transition.state[:1200],
                            "action": transition.action[:500],
                            "tool": transition.tool_name,
                            "arguments": transition.arguments_hint,
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
            return PolicyReplayResult(
                scores=[],
                method="behavioral_replay",
                error=f"{type(exc).__name__}: {exc}",
            )
