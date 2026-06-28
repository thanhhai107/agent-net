"""Workflow-neutral lifecycle helpers for the Tool Evolution module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.tool_evolution.runtime import ToolEvolutionRuntime


def write_tool_evolution_session(
    runtime: ToolEvolutionRuntime | None,
    session_dir: str,
) -> dict[str, Any]:
    """Persist per-run module state without owning the surrounding workflow."""
    if runtime is None:
        return {}
    artifact = {
        "library_id": runtime.store.library_id,
        "mode": runtime.mode.value,
        "retrieved_tools": runtime.retrieved_names,
        "created_tools": runtime.created_names,
        "capability_gaps": runtime.capability_gap_ids,
        "unverified_ephemeral_tools": sorted(
            {*runtime._ephemeral_tools, *runtime._ephemeral_generated_tools}
        ),
        "mastery_overlays": runtime.mastery_used,
        "cross_model_mastery": runtime.cross_model_mastery,
        "update_enabled": runtime.update_enabled,
    }
    path = Path(session_dir) / "tool_evolution_session.json"
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return artifact
