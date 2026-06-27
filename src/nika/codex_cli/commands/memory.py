"""Inspect, export, and reset procedural-memory experiment banks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

from agent.memory.service import ProceduralMemoryModule
from agent.memory.vector_index import QdrantMemoryIndex
from nika.config import MEMORY_DIR

memory_app = typer.Typer(help="Manage procedural-memory experiment banks.")


def _module(bank: str) -> ProceduralMemoryModule:
    return ProceduralMemoryModule(bank_id=bank)


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    return re.sub(r"(postgres(?:ql)?://[^:\s]+:)[^@\s/]+@", r"\1<redacted>@", text)


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


@memory_app.command("health")
def memory_health(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
) -> None:
    """Check PostgreSQL store connectivity and optional Qdrant readiness."""
    report = {
        "bank_id": bank,
        "store": {
            "backend": "PostgreSQLMemoryStore",
            "ready": False,
        },
        "qdrant": QdrantMemoryIndex().readiness(),
    }
    try:
        module = _module(bank)
        report["store"] = {
            "backend": type(module.store).__name__,
            "ready": True,
            "stats": module.store.bank_stats(bank),
        }
        report["qdrant"] = module.vector_index.readiness()
    except Exception as exc:
        report["store"]["reason"] = _safe_error(exc)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


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
    """Delete one experiment bank from the memory store and optional Qdrant index."""
    if not yes and not typer.confirm(f"Clear memory bank '{bank}'?", default=False):
        raise typer.Abort()
    _module(bank).clear()
    typer.echo(f"Cleared memory bank: {bank}")
