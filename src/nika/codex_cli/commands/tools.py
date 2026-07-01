"""Inspect and manage DRAFT-refined primitive tool documentation libraries."""

from __future__ import annotations

import json
import shutil

import typer

from agent.tool_evolution.store import ToolEvolutionStore
from nika.config import TOOL_EVOLUTION_DIR


tools_app = typer.Typer(help="Inspect DRAFT tool-documentation libraries.")


@tools_app.command("libraries")
def list_libraries() -> None:
    """List available DRAFT documentation libraries."""
    if not TOOL_EVOLUTION_DIR.exists():
        return
    for path in sorted(TOOL_EVOLUTION_DIR.iterdir()):
        if path.is_dir() and (path / "state.json").exists():
            stats = ToolEvolutionStore(path.name).stats()
            typer.echo(
                f"{path.name}\tdocs={stats['documents']}\t"
                f"trials={stats['trials']}\tgaps={stats['gaps']}\t"
                f"revisions={stats['revisions']}\tfrozen={stats['frozen_documents']}"
            )


@tools_app.command("show")
def show_library(
    library_id: str = typer.Argument(..., help="Tool library id."),
) -> None:
    """Print one DRAFT library as JSON."""
    state = ToolEvolutionStore(library_id).load()
    typer.echo(state.model_dump_json(indent=2))


@tools_app.command("reset")
def reset_library(
    library_id: str = typer.Argument(..., help="Tool library id."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation."),
) -> None:
    """Delete one persistent tool library."""
    store = ToolEvolutionStore(library_id)
    if not store.library_dir.exists():
        raise typer.BadParameter(f"Tool library does not exist: {store.library_id}")
    if not yes and not typer.confirm(f"Delete tool library '{store.library_id}'?"):
        raise typer.Abort()
    shutil.rmtree(store.library_dir)
    typer.echo(json.dumps({"deleted": store.library_id}))
