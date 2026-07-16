"""Inspect, export, and reset Skill-Pro procedural-skill banks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from agent.module_config import module_defaults
from agent.extensions.config import (
    DEFAULT_LLM_PROVIDER as DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
from nika.utils.agent_config import resolve_max_steps
from agent.procedural_memory.service import ProceduralMemoryModule
from agent.extensions.config import PROCEDURAL_MEMORY_DIR
from nika.config import BENCHMARK_DIR
from nika.extensions.benchmark import load_custom_benchmark
from nika.workflows.benchmark.load_config import load_benchmark_evolve_first_cases

procedural_memory_app = typer.Typer(help="Manage Skill-Pro Procedural Memory banks.")
PROCEDURAL_MEMORY_DEFAULTS = module_defaults().procedural_memory


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
    payload: dict[str, object] = {"cases": rows}
    evolve_first_cases = load_benchmark_evolve_first_cases(source)
    if evolve_first_cases is not None:
        payload["evolve_first_cases"] = min(evolve_first_cases, len(rows))
    target.write_text(
        yaml.dump(payload, sort_keys=False, allow_unicode=True),
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
    evolve_until: int | None = typer.Option(
        None,
        "--evolve-until",
        min=0,
        help="Index Evaluate: evolve the first N cases, then evaluate read-only. Use 0 with --keep-bank to evaluate an existing bank.",
    ),
    bank: str = typer.Option(
        "procedural-memory-smoke", "--bank", help="Procedural Memory bank id."
    ),
    reset_bank: bool = typer.Option(
        True,
        "--reset-bank/--keep-bank",
        help="Clear the bank before running; use --keep-bank only to resume deliberately.",
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
    tokens: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.token_budget,
        "--token-budget",
        "--tokens",
        min=100,
        help="Token budget for the bounded active-skill policy context.",
    ),
    max_skill_age: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.max_skill_age,
        "--max-skill-age",
        min=1,
        help="Maximum primitive actions controlled by one active skill.",
    ),
    pool_size: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.pool_size,
        "--pool-size",
        min=1,
        help="Maximum active skill-pool size.",
    ),
    update_threshold: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.evolution_threshold,
        "--update-threshold",
        min=2,
        help="Parent trajectories required for one semantic-gradient update.",
    ),
    best_of_n: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.best_of_n,
        "--best-of-n",
        min=1,
        help="Independent candidates evaluated by the configured verification gate.",
    ),
    ppo_epsilon: float = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.ppo_epsilon,
        "--ppo-epsilon",
        min=0.0,
        max=1.0,
        help="PPO-style surrogate importance-ratio clipping epsilon.",
    ),
    selection_epsilon: float = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon,
        "--selection-epsilon",
        min=0.0,
        max=1.0,
        help="Initial deterministic epsilon-greedy skill-selection rate.",
    ),
    experience_pool_size: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.experience_pool_size,
        "--experience-pool-size",
        min=1,
        help="Maximum persisted replay experiences.",
    ),
    baseline_ema_alpha: float = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.baseline_ema_alpha,
        "--baseline-ema-alpha",
        min=0.01,
        max=1.0,
        help="EMA update weight for scenario reward baselines.",
    ),
    selection_epsilon_decay_cases: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.selection_epsilon_decay_cases,
        "--selection-epsilon-decay-cases",
        min=1,
        help="Cases over which selection epsilon decays toward 0.05.",
    ),
    acceptance_margin: float = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.acceptance_margin,
        "--acceptance-margin",
        min=0.0,
        help="Minimum verification surrogate improvement.",
    ),
    verifier: str = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.verifier,
        "--verifier",
        help="Candidate verifier: behavioral_replay, structured_replay, or policy_logprob.",
    ),
    holdout_size: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.holdout_size,
        "--holdout-size",
        min=1,
        help="Disjoint verification trajectories reserved from each evolution batch.",
    ),
    min_positive_advantage: int = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.min_positive_advantage,
        "--min-positive-advantage",
        min=0,
        help="Positive-advantage holdouts required by replay verifiers.",
    ),
    evolver_model: str = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.llm_model,
        "--evolver-model",
        help="Semantic-gradient and candidate-generation model.",
    ),
    policy_scorer_model: str = typer.Option(
        PROCEDURAL_MEMORY_DEFAULTS.skill_logprob_model,
        "--policy-scorer-model",
        help="Behavioral or log-prob verification model.",
    ),
) -> None:
    """Run a Skill-Pro Procedural Memory benchmark with concise defaults."""
    if not file.exists():
        raise typer.BadParameter(f"YAML does not exist: {file}")
    supported_verifiers = {
        "behavioral_replay",
        "structured_replay",
        "policy_logprob",
    }
    if verifier not in supported_verifiers:
        raise typer.BadParameter(
            "--verifier must be behavioral_replay, structured_replay, or policy_logprob"
        )
    if holdout_size >= update_threshold:
        raise typer.BadParameter(
            "--holdout-size must be smaller than --update-threshold"
        )
    if min_positive_advantage > holdout_size:
        raise typer.BadParameter(
            "--min-positive-advantage must not exceed --holdout-size"
        )
    if evolve_until == 0 and reset_bank:
        raise typer.BadParameter(
            "--evolve-until 0 evaluates an existing bank; pass --keep-bank to avoid clearing it"
        )

    if reset_bank:
        _module(bank).clear()
        typer.echo(
            f"Reset Procedural Memory bank and rebuilt Skill-Pro seed pool: {bank}"
        )
    resolved_max_steps = resolve_max_steps(max_steps)

    selected_yaml = _limited_yaml_path(file, limit=limit, bank=bank)
    typer.echo(
        "Running Procedural Memory benchmark: "
        f"yaml={selected_yaml} bank={bank} index_evaluate={evolve_until} "
        f"backend={llm_backend} model={model}"
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
        "--procedural-memory",
        bank,
        "--procedural-memory-token-budget",
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
        "--procedural-memory-experience-pool-size",
        str(experience_pool_size),
        "--procedural-memory-baseline-ema-alpha",
        str(baseline_ema_alpha),
        "--procedural-memory-selection-epsilon-decay-cases",
        str(selection_epsilon_decay_cases),
        "--procedural-memory-acceptance-margin",
        str(acceptance_margin),
        "--procedural-memory-verifier",
        verifier,
        "--procedural-memory-holdout-size",
        str(holdout_size),
        "--procedural-memory-min-positive-advantage",
        str(min_positive_advantage),
        "--procedural-memory-evolver-model",
        evolver_model,
        "--procedural-memory-policy-scorer-model",
        policy_scorer_model,
    ]
    if evolve_until is not None:
        command.extend(["--evolve-until", str(evolve_until)])
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
            _module(bank).store.bank_stats(),
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
            "stats": module.store.bank_stats(),
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
    path = _module(bank).snapshot(output_path=target)
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
