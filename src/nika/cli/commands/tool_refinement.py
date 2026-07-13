"""Inspect and manage DRAFT-refined primitive tool documentation libraries."""

from __future__ import annotations

import json

import typer

from agent.tool_refinement.store import ToolRefinementStore
from agent.extensions.config import TOOL_REFINEMENT_DIR


tools_app = typer.Typer(help="Inspect DRAFT Tool Refinement libraries.")


@tools_app.command("libraries")
def list_libraries() -> None:
    """List available DRAFT Tool Refinement libraries."""
    if not TOOL_REFINEMENT_DIR.exists():
        return
    for path in sorted(TOOL_REFINEMENT_DIR.iterdir()):
        if path.is_dir() and (path / "state.json").exists():
            stats = ToolRefinementStore(path.name).stats()
            typer.echo(
                f"{path.name}\tdocs={stats['documents']}\t"
                f"trials={stats['trials']}\texplore={stats['explorations']}\t"
                f"analyze={stats['analyzer_suggestions']}\tgaps={stats['gaps']}\t"
                f"revisions={stats['revisions']}\tfrozen={stats['frozen_documents']}\t"
                f"llm_fail={stats['llm_failures']}\t"
                f"mastered={stats['mastered_tools']}\t"
                f"doc_path={stats['avg_documented_path_rate']:.2f}\t"
                f"success_path={stats['avg_success_path_rate']:.2f}"
            )


@tools_app.command("show")
def show_library(
    library_id: str = typer.Argument(..., help="Tool library id."),
) -> None:
    """Print one DRAFT Tool Refinement library as JSON."""
    state = ToolRefinementStore(library_id).load()
    typer.echo(state.model_dump_json(indent=2))


@tools_app.command("reset")
def reset_library(
    library_id: str = typer.Argument(..., help="Tool library id."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation."),
) -> None:
    """Delete one persistent tool library."""
    store = ToolRefinementStore(library_id)
    if not store.state_path.exists():
        raise typer.BadParameter(f"Tool library does not exist: {store.library_id}")
    if not yes and not typer.confirm(f"Delete tool library '{store.library_id}'?"):
        raise typer.Abort()
    store.clear()
    typer.echo(json.dumps({"deleted": store.library_id}))


if __name__ == "__main__":
    tools_app()
