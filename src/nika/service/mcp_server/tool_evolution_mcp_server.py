"""FastMCP adapter exposing DRAFT-refined primitive tool documentation."""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from agent.tool_evolution.store import ToolEvolutionStore
from nika.utils.errors import safe_tool

mcp = FastMCP("nika_diagnostic_tool_docs")


def _store() -> ToolEvolutionStore:
    library_id = os.environ.get("NIKA_TOOL_LIBRARY_ID", "").strip()
    if not library_id:
        raise ValueError("NIKA_TOOL_LIBRARY_ID is required")
    return ToolEvolutionStore(library_id)


@safe_tool
@mcp.tool()
def list_refined_tool_docs(include_frozen: bool = True) -> list[dict[str, Any]]:
    """List DRAFT-refined documentation for primitive diagnostic tools."""
    state = _store().load()
    docs = []
    for doc in state.documents.values():
        if doc.frozen and not include_frozen:
            continue
        docs.append(
            {
                "name": doc.name,
                "version": doc.version,
                "frozen": doc.frozen,
                "tool_usage_description": doc.tool_usage_description,
                "description": doc.refined_description(),
                "parameters": {
                    key: value.model_dump() for key, value in doc.parameters.items()
                },
                "usage_notes": doc.usage_notes,
                "failure_modes": doc.failure_modes,
                "exploration_suggestions": doc.exploration_suggestions,
                "mastery_score": doc.mastery_score,
                "last_convergence_score": doc.last_convergence_score,
            }
        )
    return sorted(docs, key=lambda item: item["name"])


@safe_tool
@mcp.tool()
def get_refined_tool_doc(tool_name: str) -> dict[str, Any]:
    """Return one refined primitive-tool document."""
    doc = _store().get_document(tool_name)
    if doc is None:
        raise ValueError(f"No refined documentation for primitive tool: {tool_name}")
    return doc.model_dump()


if __name__ == "__main__":
    mcp.run(transport="stdio")
