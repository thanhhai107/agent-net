"""Small runtime artifact writer for DRAFT Tool Refinement."""

from __future__ import annotations

import json
from pathlib import Path

from agent.tool_refinement.runtime import ToolRefinementRuntime


def write_tool_refinement_session(
    runtime: ToolRefinementRuntime | None,
    session_dir: str | Path,
) -> Path | None:
    if runtime is None:
        return None
    path = Path(session_dir) / "tool_refinement_session.json"
    path.write_text(
        json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
