"""Runtime injection for DRAFT-refined primitive tool documentation."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from agent.tool_evolution.models import ToolDocumentation
from agent.tool_evolution.store import ToolEvolutionStore


class ToolEvolutionRuntime:
    """Inject refined documentation while keeping the primitive tool surface fixed."""

    def __init__(
        self,
        *,
        session: Any,
        primitive_tools: list[BaseTool],
        library_id: str,
        model: str = "",
        task_description: str = "",
    ) -> None:
        self.session = session
        self.primitive_tools = list(primitive_tools)
        self.library_id = library_id
        self.model = model
        self.task_description = task_description
        self.store = ToolEvolutionStore(library_id)
        self._docs = self.store.load().documents

    def build_tools(self) -> list[BaseTool]:
        """Return the same primitive tools with DRAFT docs appended to descriptions."""
        for tool in self.primitive_tools:
            doc = self._docs.get(tool.name)
            if doc is None:
                continue
            refined = doc.refined_description()
            original = getattr(tool, "description", "") or ""
            if refined and refined not in original:
                tool.description = (
                    f"{original.strip()}\n\nDRAFT refined guidance:\n{refined}"
                )
        return self.primitive_tools

    def prompt_suffix(self) -> str:
        if not self._docs:
            return ""
        active_docs = [doc for doc in self._docs.values() if not doc.frozen]
        if not active_docs:
            active_docs = list(self._docs.values())
        state = self.store.load()
        snippets = []
        for doc in sorted(active_docs, key=lambda item: item.name)[:8]:
            snippets.append(f"- {doc.name}: {doc.refined_description(max_chars=600)}")
        return (
            "\n\nDRAFT tool documentation memory:\n"
            "The primitive tool surface is fixed. Use the following refined docs "
            "to choose valid arguments, avoid known failure modes, and follow "
            "DRAFT next-check suggestions when more evidence is needed.\n"
            + (
                f"{state.library_usage_description}\n"
                if state.library_usage_description
                else ""
            )
            + "\n".join(snippets)
        )

    def snapshot(self) -> dict[str, Any]:
        state = self.store.load()
        return {
            "library_id": self.library_id,
            "model": self.model,
            "task_description": self.task_description,
            "available_documents": sorted(self._docs),
            "library_usage_description": state.library_usage_description,
            "tool_stats": {
                name: stat.model_dump(mode="json")
                for name, stat in sorted(state.tool_stats.items())
            },
            "explorations": len(state.explorations),
            "analyzer_suggestions": len(state.analyzer_suggestions),
            "primitive_tools": [tool.name for tool in self.primitive_tools],
        }


def make_document_from_tool(tool: BaseTool) -> ToolDocumentation:
    description = getattr(tool, "description", "") or ""
    return ToolDocumentation(name=tool.name, description=description.strip())
