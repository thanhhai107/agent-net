"""Inspect, export, and reset procedural-memory experiment banks."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from agent.memory.service import HybridMemoryModule
from nika.config import MEMORY_DIR

memory_app = typer.Typer(help="Manage procedural-memory experiment banks.")


def _module(bank: str) -> HybridMemoryModule:
    return HybridMemoryModule(bank_id=bank)


@memory_app.command("inspect")
def memory_inspect(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
) -> None:
    """Print memory, episode, and retrieval counts for one bank."""
    typer.echo(
        json.dumps(
            _module(bank).store.bank_stats(bank),
            ensure_ascii=False,
            indent=2,
        )
    )


@memory_app.command("snapshot")
def memory_snapshot(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSONL path.",
    ),
) -> None:
    """Export a reproducible JSONL snapshot of one bank."""
    target = output or (Path(MEMORY_DIR) / f"{bank}.snapshot.jsonl")
    path = _module(bank).snapshot(session_id="manual", output_path=target)
    typer.echo(f"Wrote memory snapshot: {path}")


@memory_app.command("clear")
def memory_clear(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation."),
) -> None:
    """Delete one experiment bank from SQLite and its optional Qdrant index."""
    if not yes and not typer.confirm(f"Clear memory bank '{bank}'?", default=False):
        raise typer.Abort()
    _module(bank).clear()
    typer.echo(f"Cleared memory bank: {bank}")
