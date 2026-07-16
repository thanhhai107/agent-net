"""Canonical Skill-MDP policy context shared by execution and replay."""

from __future__ import annotations

import json
from typing import Any

from agent.procedural_memory.models import ProceduralSkill
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT


def serialize_primitive_action(tool_name: str, arguments: dict[str, Any]) -> str:
    """Serialize a structured tool call deterministically for teacher forcing."""

    payload = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{tool_name}({payload})" if tool_name else payload


def build_skill_policy_suffix(
    state: str,
    skill: ProceduralSkill | None,
    *,
    max_tokens: int | None = None,
    include_state: bool = True,
) -> str:
    budget_chars = max(0, int(max_tokens or 0)) * 4
    state_section = "\n\n[CURRENT OBSERVABLE STATE]\n" if include_state else ""
    skill_header = "\n\n[ACTIVE SKILL-MDP OPTION]\n"
    guidance = (
        "\n\nUse the option as procedural guidance, not as evidence. Option "
        "termination only returns control to the skill selector; it neither proves "
        "the diagnosis nor authorizes submission. Use current-run observations only. "
        "If evidence is incomplete, choose the next diagnostic action or tool call. "
        "If evidence already supports a complete diagnosis, stop calling tools and "
        "return a concise diagnosis report with anomaly status and any supported "
        "localization and root cause. Do not submit from the diagnosis phase.\n"
    )
    # Reserve a small truncation margin so the rendered suffix never exceeds
    # the caller's context allowance.
    fixed_chars = len(state_section) + len(skill_header) + len(guidance) + 12
    variable_chars = max(0, budget_chars - fixed_chars) if budget_chars else 0
    state_limit = variable_chars // 2 if include_state and variable_chars else 0
    # Runtime messages already carry the observable state. Keep the same half-
    # budget previously available to the Skill while dropping that duplicate.
    skill_limit = (
        variable_chars - state_limit
        if include_state
        else variable_chars // 2
        if variable_chars
        else 0
    )
    state_text = state
    if state_limit and len(state_text) > state_limit:
        state_text = state_text[:state_limit] + "..."
    skill_text = skill.format_for_llm() if skill else "No active procedural skill."
    if skill_limit and len(skill_text) > skill_limit:
        skill_text = skill_text[:skill_limit] + "..."
    rendered_state = state_section + f"{state_text}\n" if include_state else ""
    suffix = rendered_state + skill_header + f"{skill_text}\n" + guidance
    if budget_chars and len(suffix) > budget_chars:
        # Final guard for unusually small budgets where static instructions dominate.
        return suffix[:budget_chars]
    return suffix


def build_skill_policy_prefix(
    state: str,
    skill: ProceduralSkill | None,
) -> str:
    return OVERALL_DIAGNOSIS_PROMPT + build_skill_policy_suffix(state, skill)


def build_runtime_skill_policy_prefix(
    skill: ProceduralSkill | None,
    *,
    max_tokens: int,
) -> str:
    """Render the exact system prompt installed by the Skill runtime.

    Observable state and chat/tool messages are carried separately by LangGraph;
    this helper deliberately reproduces only the system-prompt portion that the
    completions log-prob endpoint can replay.
    """

    return OVERALL_DIAGNOSIS_PROMPT + build_skill_policy_suffix(
        "",
        skill,
        max_tokens=max_tokens,
        include_state=False,
    )
