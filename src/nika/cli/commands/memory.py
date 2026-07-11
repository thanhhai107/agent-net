"""Inspect, export, and reset Skill-Pro procedural-skill banks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
from nika.utils.agent_config import resolve_max_steps
from agent.memory.service import ProceduralMemoryModule
from agent.extensions.config import MEMORY_DIR
from nika.config import BENCHMARK_DIR
from nika.extensions.benchmark import load_custom_benchmark

memory_app = typer.Typer(help="Manage Skill-Pro procedural-skill banks.")


def _module(bank: str) -> ProceduralMemoryModule:
    return ProceduralMemoryModule(bank_id=bank)


def _safe_error(exc: Exception) -> str:
    return str(exc)


def _limited_yaml_path(source: Path, *, limit: int | None, bank: str) -> Path:
    if limit is None:
        return source
    if limit < 1:
        raise typer.BadParameter("--limit must be >= 1")
    rows = load_custom_benchmark(source)[:limit]
    if not rows:
        raise typer.BadParameter(f"{source} has no benchmark cases")
    safe_bank = re.sub(r"[^A-Za-z0-9_.-]+", "_", bank).strip("._") or "default"
    target_dir = Path(MEMORY_DIR) / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_bank}.first-{len(rows)}.yaml"
    target.write_text(
        yaml.dump({"cases": rows}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


@memory_app.command("run")
def memory_run(
    file: Path = typer.Option(
        BENCHMARK_DIR / "benchmark_test.yaml",
        "-f",
        "--file",
        help="Shared memory/tool-evolution benchmark YAML. Defaults to benchmark/benchmark_test.yaml.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Run only the first N rows for a quick memory-only smoke test.",
    ),
    bank: str = typer.Option("memory-smoke", "--bank", help="Memory-bank id."),
    read: bool = typer.Option(
        False,
        "--read",
        help="Read the bank without updating it.",
    ),
    reset_bank: bool = typer.Option(
        False,
        "--reset-bank/--keep-bank",
        help="Clear the bank before running.",
    ),
    llm_backend: str = typer.Option(
        DEFAULT_LLM_BACKEND,
        "-b",
        "--backend",
        help="LLM provider.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "-m", "--model", help="Model id."),
    max_steps: int | None = typer.Option(
        None,
        "-n",
        "--max-steps",
        help="Per-worker step limit. Defaults to NIKA_MAX_STEPS.",
    ),
    k: int = typer.Option(
        5,
        "-k",
        "--k",
        min=1,
        max=20,
        help="Maximum procedural skills injected into one diagnosis.",
    ),
    tokens: int = typer.Option(
        1500,
        "--tokens",
        min=100,
        help="Estimated token budget for retrieved skills.",
    ),
) -> None:
    """Run a Skill-Pro memory-only benchmark stream with concise defaults."""
    mode = "read" if read else "evolve"
    if not file.exists():
        raise typer.BadParameter(f"YAML does not exist: {file}")

    if reset_bank:
        _module(bank).clear()
        typer.echo(f"Reset memory bank and rebuilt Skill-Pro seed pool: {bank}")
    resolved_max_steps = resolve_max_steps(max_steps)

    selected_yaml = _limited_yaml_path(file, limit=limit, bank=bank)
    typer.echo(
        "Running memory-only benchmark: "
        f"yaml={selected_yaml} bank={bank} mode={mode} "
        f"backend={llm_backend} model={model}"
    )
    memory_flag = "--memory-read" if read else "--memory"
    command = [
        sys.executable,
        "-m",
        "nika.extensions.benchmark",
        "--config",
        str(selected_yaml),
        "--provider",
        llm_backend,
        "--model",
        model,
        "--max-steps",
        str(resolved_max_steps),
        memory_flag,
        bank,
        "--memory-k",
        str(k),
        "--memory-tokens",
        str(tokens),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc


@memory_app.command("inspect")
def memory_inspect(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
) -> None:
    """Print skill, episode, and PPO decision counts for one bank."""
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
    """Check local JSON skill store readiness."""
    report = {
        "bank_id": bank,
        "store": {
            "backend": "SkillMemoryStore",
            "ready": False,
        },
    }
    try:
        module = _module(bank)
        report["store"] = {
            "backend": type(module.store).__name__,
            "ready": True,
            "stats": module.store.bank_stats(bank),
        }
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
    """Export a reproducible JSONL snapshot of one skill bank."""
    target = output or (Path(MEMORY_DIR) / f"{bank}.snapshot.jsonl")
    path = _module(bank).snapshot(session_id="manual", output_path=target)
    typer.echo(f"Wrote memory snapshot: {path}")


@memory_app.command("clear")
def memory_clear(
    bank: str = typer.Option("default", "--bank", help="Memory-bank id."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation."),
) -> None:
    """Delete one experiment bank from the local JSON skill store."""
    if not yes and not typer.confirm(f"Clear memory bank '{bank}'?", default=False):
        raise typer.Abort()
    _module(bank).clear()
    typer.echo(f"Reset memory bank and rebuilt Skill-Pro seed pool: {bank}")


if __name__ == "__main__":
    memory_app()
