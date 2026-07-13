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
) -> str:
    skill_text = skill.format_for_llm() if skill else "No active procedural skill."
    return (
        "\n\n[CURRENT OBSERVABLE STATE]\n"
        f"{state}\n\n"
        "[ACTIVE SKILL-MDP OPTION]\n"
        f"{skill_text}\n\n"
        "Use the option as procedural guidance, not as evidence. Generate the "
        "next primitive diagnostic action from current-run observations only.\n\n"
        "Return the next primitive action or tool call only.\nAction:\n"
    )


def build_skill_policy_prefix(
    state: str,
    skill: ProceduralSkill | None,
) -> str:
    return OVERALL_DIAGNOSIS_PROMPT + build_skill_policy_suffix(state, skill)
