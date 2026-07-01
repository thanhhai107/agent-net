"""Small runtime artifact writer for DRAFT tool evolution."""

from __future__ import annotations

import json
from pathlib import Path

from agent.tool_evolution.runtime import ToolEvolutionRuntime


def write_tool_evolution_session(
    runtime: ToolEvolutionRuntime | None,
    session_dir: str | Path,
) -> Path | None:
    if runtime is None:
        return None
    path = Path(session_dir) / "tool_evolution_session.json"
    path.write_text(
        json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
