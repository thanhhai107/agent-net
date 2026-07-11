"""Current-run tool evidence extraction shared by diagnosis and submission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


INTEGRATED_GUIDANCE_MARKER = "[Integrated learning guidance - not evidence]"


@dataclass(frozen=True)
class ToolObservation:
    """One current-run tool observation eligible for final evidence review."""

    tool: str = ""
    tool_input: str = ""
    summary: str = ""


def observations_from_runtime_snapshot(
    snapshot: dict[str, Any] | None,
) -> list[ToolObservation]:
    """Build evidence records from the Skill-Pro runtime transition log."""
    if not snapshot:
        return []
    observations: list[ToolObservation] = []
    for item in snapshot.get("recent_transitions") or []:
        if not isinstance(item, dict):
            continue
        observations.append(
            ToolObservation(
                tool=str(item.get("tool") or ""),
                tool_input=_compact(item.get("tool_input")),
                summary=_strip_learning_guidance(item.get("observation_summary")),
            )
        )
    return observations


def observations_from_messages(messages: Sequence[Any] | None) -> list[ToolObservation]:
    """Extract evidence-bearing tool messages from a LangChain trajectory."""
    if not messages:
        return []
    observations: list[ToolObservation] = []
    pending_calls: dict[str, tuple[str, Any]] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            if call_id and name:
                pending_calls[call_id] = (
                    name,
                    call.get("args", call.get("arguments", {})),
                )
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        if not tool_call_id and getattr(message, "type", "") != "tool":
            continue
        pending_name, pending_input = pending_calls.get(tool_call_id, ("", None))
        observations.append(
            ToolObservation(
                tool=str(
                    getattr(message, "name", "") or pending_name
                ),
                tool_input=_compact(pending_input),
                summary=_strip_learning_guidance(getattr(message, "content", "")),
            )
        )
    return observations


def _strip_learning_guidance(value: Any) -> str:
    text = str(value or "")
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    return text.strip()


def _compact(value: Any, *, limit: int = 500) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
