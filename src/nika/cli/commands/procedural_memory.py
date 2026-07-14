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
from agent.procedural_memory.service import ProceduralMemoryModule
from agent.extensions.config import PROCEDURAL_MEMORY_DIR
from nika.config import BENCHMARK_DIR
from nika.extensions.benchmark import load_custom_benchmark

procedural_memory_app = typer.Typer(help="Manage Skill-Pro Procedural Memory banks.")


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
    target_dir = Path(PROCEDURAL_MEMORY_DIR) / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_bank}.first-{len(rows)}.yaml"
    target.write_text(
        yaml.dump({"cases": rows}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


@procedural_memory_app.command("run")
def procedural_memory_run(
    file: Path = typer.Option(
        BENCHMARK_DIR / "benchmark_evolve.yaml",
        "-f",
        "--file",
        help="Shared Procedural Memory/Tool Refinement benchmark YAML. Defaults to benchmark/benchmark_evolve.yaml.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Run only the first N rows for a quick Procedural Memory smoke test.",
    ),
    bank: str = typer.Option(
        "procedural-memory-smoke", "--bank", help="Procedural Memory bank id."
    ),
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
    max_skill_age: int = typer.Option(
        8,
        "--max-skill-age",
        min=1,
        help="Maximum primitive actions controlled by one active skill.",
    ),
    pool_size: int = typer.Option(
        32,
        "--pool-size",
        min=1,
        help="Maximum active skill-pool size.",
    ),
    update_threshold: int = typer.Option(
        6,
        "--update-threshold",
        min=1,
        help="Parent trajectories required for one semantic-gradient update.",
    ),
    best_of_n: int = typer.Option(
        3,
        "--best-of-n",
        min=1,
        help="Independent candidates evaluated by the PPO gate.",
    ),
    ppo_epsilon: float = typer.Option(
        0.2,
        "--ppo-epsilon",
        min=0.0,
        max=1.0,
        help="PPO-style importance-ratio clipping epsilon.",
    ),
    selection_epsilon: float = typer.Option(
        0.3,
        "--selection-epsilon",
        min=0.0,
        max=1.0,
        help="Initial deterministic epsilon-greedy skill-selection rate.",
    ),
) -> None:
    """Run a Skill-Pro Procedural Memory benchmark with concise defaults."""
    mode = "read" if read else "evolve"
    if not file.exists():
        raise typer.BadParameter(f"YAML does not exist: {file}")

    if reset_bank:
        _module(bank).clear()
        typer.echo(
            f"Reset Procedural Memory bank and rebuilt Skill-Pro seed pool: {bank}"
        )
    resolved_max_steps = resolve_max_steps(max_steps)

    selected_yaml = _limited_yaml_path(file, limit=limit, bank=bank)
    typer.echo(
        "Running Procedural Memory benchmark: "
        f"yaml={selected_yaml} bank={bank} mode={mode} "
        f"backend={llm_backend} model={model}"
    )
    procedural_memory_flag = (
        "--procedural-memory-read" if read else "--procedural-memory"
    )
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
        procedural_memory_flag,
        bank,
        "--procedural-memory-k",
        str(k),
        "--procedural-memory-tokens",
        str(tokens),
        "--procedural-memory-max-skill-age",
        str(max_skill_age),
        "--procedural-memory-pool-size",
        str(pool_size),
        "--procedural-memory-update-threshold",
        str(update_threshold),
        "--procedural-memory-best-of-n",
        str(best_of_n),
        "--procedural-memory-ppo-epsilon",
        str(ppo_epsilon),
        "--procedural-memory-selection-epsilon",
        str(selection_epsilon),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc


@procedural_memory_app.command("inspect")
def procedural_memory_inspect(
    bank: str = typer.Option("default", "--bank", help="Procedural Memory bank id."),
) -> None:
    """Print skill, episode, and PPO decision counts for one bank."""
    typer.echo(
        json.dumps(
            _module(bank).store.bank_stats(bank),
            ensure_ascii=False,
            indent=2,
        )
    )


@procedural_memory_app.command("health")
def procedural_memory_health(
    bank: str = typer.Option("default", "--bank", help="Procedural Memory bank id."),
) -> None:
    """Check local JSON skill store readiness."""
    report = {
        "bank_id": bank,
        "store": {
            "backend": "ProceduralMemoryStore",
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


@procedural_memory_app.command("snapshot")
def procedural_memory_snapshot(
    bank: str = typer.Option("default", "--bank", help="Procedural Memory bank id."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output JSONL path.",
    ),
) -> None:
    """Export a reproducible JSONL snapshot of one skill bank."""
    target = output or (Path(PROCEDURAL_MEMORY_DIR) / f"{bank}.snapshot.jsonl")
    path = _module(bank).snapshot(session_id="manual", output_path=target)
    typer.echo(f"Wrote Procedural Memory snapshot: {path}")


@procedural_memory_app.command("clear")
def procedural_memory_clear(
    bank: str = typer.Option("default", "--bank", help="Procedural Memory bank id."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation."),
) -> None:
    """Delete one experiment bank from the local JSON skill store."""
    if not yes and not typer.confirm(
        f"Clear Procedural Memory bank '{bank}'?", default=False
    ):
        raise typer.Abort()
    _module(bank).clear()
    typer.echo(f"Reset Procedural Memory bank and rebuilt Skill-Pro seed pool: {bank}")


if __name__ == "__main__":
    procedural_memory_app()
