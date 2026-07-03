"""Configuration helpers for sdk.codex_sdk (local ~/.codex/auth.json)."""

from __future__ import annotations

from pathlib import Path

from agent.local_cli.codex_cli.codex_worker import REASONING_EFFORT_LEVELS


def codex_sdk_local_auth_available() -> bool:
    """True when local Codex auth exists (``codex login``)."""
    return (Path.home() / ".codex" / "auth.json").is_file()


def validate_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    if reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise ValueError(
            f"reasoning_effort must be one of {REASONING_EFFORT_LEVELS}, got {reasoning_effort!r}"
        )
    return reasoning_effort
