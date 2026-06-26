"""Benchmark runner: full pipeline from env to evaluation."""

from pathlib import Path

import typer

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.tool_evolution.models import ToolEvolutionMode
from nika.net_env.net_env_pool import scenario_requires_topo_tier
from nika.workflows.benchmark.run import (
    default_benchmark_csv_path,
    run_benchmark_from_csv,
    run_single_benchmark,
)

benchmark_app = typer.Typer(
    help="Run curated benchmark cases (env → fault → agent → eval)."
)


@benchmark_app.command("run")
def benchmark_run(
    scenario: str | None = typer.Argument(
        default=None,
        metavar="SCENARIO",
        help="Scenario id for a single case (omit for CSV batch mode).",
    ),
    csv: Path | None = typer.Option(
        None,
        "--csv",
        help="Benchmark CSV path (batch mode). Defaults to benchmark/benchmark_selected.csv under the repo root.",
    ),
    problem: str | None = typer.Option(
        None,
        "--problem",
        help="Problem id for a single case (required with SCENARIO).",
    ),
    tier: str | None = typer.Option(
        None,
        "-t",
        "--tier",
        help="Topology tier s, m, or l (required only for scalable scenarios).",
    ),
    agent_type: str = typer.Option(
        "react", "-a", "--agent", help="Agent implementation."
    ),
    llm_backend: str = typer.Option(
        DEFAULT_LLM_BACKEND,
        "-b",
        "--backend",
        help="LLM provider (openai, ollama, deepseek, netmind).",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "-m", "--model", help="Model id for the agent."
    ),
    max_steps: int = typer.Option(
        20,
        "-n",
        "--max-steps",
        help=(
            "Per-worker step limit for LangGraph agents; also the maximum "
            "executed plan items for plan-execute. Ignored for cli."
        ),
    ),
    max_attempts: int = typer.Option(
        3,
        "-r",
        "--max-attempts",
        min=1,
        help="Maximum attempts for the reflexion agent; ignored by other agents.",
    ),
    parallel: int = typer.Option(
        1,
        "-j",
        "--parallel",
        help="Number of benchmark cases to run concurrently in CSV batch mode (default: 1).",
    ),
    memory_mode: str = typer.Option(
        "off",
        "--memory-mode",
        help="Composable memory module: off, read, or evolve.",
    ),
    memory_bank: str = typer.Option(
        "default",
        "--memory-bank",
        help="Persistent memory-bank id when memory is enabled.",
    ),
    memory_top_k: int = typer.Option(
        5,
        "--memory-top-k",
        min=1,
        max=20,
        help="Maximum memories injected into one diagnosis.",
    ),
    memory_token_budget: int = typer.Option(
        1500,
        "--memory-token-budget",
        min=100,
        help="Maximum estimated tokens used by retrieved memory.",
    ),
    run_judge: bool = typer.Option(
        False,
        "--judge",
        help="Run LLM-as-judge after metrics (default: metrics and publish only).",
    ),
    judge_backend: str | None = typer.Option(
        None,
        "--judge-backend",
        help="LLM provider for the judge (defaults to the global LLM backend when --judge is set).",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Model id for the judge (defaults to the global LLM model when --judge is set).",
    ),
    oracle_routing: bool = typer.Option(
        False,
        "--oracle-routing",
        help="Use hidden problem labels for MCP server selection (oracle baseline).",
    ),
    tool_evolution: bool = typer.Option(
        False,
        "--tool-evolution/--no-tool-evolution",
        help="Enable Tool Evolution as a module for the selected workflow.",
    ),
    tool_library: str | None = typer.Option(
        None,
        "--tool-library",
        help="Persistent Tool Evolution library id; generated in batch mode when omitted.",
    ),
    evolution_mode: str = typer.Option(
        ToolEvolutionMode.DUAL.value,
        "--evolution-mode",
        help="Tool Evolution mode: mastery, distill, dual, dual-no-validation, dual-no-dedup.",
    ),
) -> None:
    """Run one benchmark row from CSV, or a single case when SCENARIO and --problem are set."""
    if not run_judge and (judge_backend is not None or judge_model is not None):
        raise typer.BadParameter(
            "Pass --judge to enable LLM judge; omit --judge-backend/--judge-model otherwise."
        )
    judge_backend = judge_backend or DEFAULT_LLM_BACKEND
    judge_model = judge_model or DEFAULT_MODEL
    try:
        ToolEvolutionMode(evolution_mode)
    except ValueError as exc:
        raise typer.BadParameter(
            "evolution_mode must be one of "
            + ", ".join(item.value for item in ToolEvolutionMode)
        ) from exc
    if memory_mode not in {"off", "read", "evolve"}:
        raise typer.BadParameter("--memory-mode must be off, read, or evolve")
    if memory_mode != "off" and agent_type.lower() not in {
        "react",
        "plan-execute",
        "reflexion",
    }:
        raise typer.BadParameter(
            "memory is supported only for: react, plan-execute, reflexion"
        )
    if memory_mode == "evolve" and parallel != 1:
        raise typer.BadParameter("online memory evolution requires --parallel 1")

    if scenario is not None and csv is not None:
        raise typer.BadParameter(
            "Use either SCENARIO (single-case mode) or --csv (batch mode), not both."
        )

    single_mode = scenario is not None

    if single_mode:
        if parallel != 1:
            raise typer.BadParameter(
                "--parallel applies to CSV batch mode only; omit it for a single case."
            )
        if not problem:
            raise typer.BadParameter("--problem is required when SCENARIO is given.")
        if scenario_requires_topo_tier(scenario) and not tier:
            raise typer.BadParameter(
                f"Scenario '{scenario}' requires -t/--tier (s, m, or l)."
            )
        if not scenario_requires_topo_tier(scenario) and tier is not None:
            raise typer.BadParameter(
                f"Scenario '{scenario}' does not use tiers; omit -t/--tier."
            )
        topo = tier or ""
        run_single_benchmark(
            problem=problem,
            scenario=scenario,
            topo_size=topo,
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            memory_mode=memory_mode,
            memory_bank=memory_bank,
            memory_top_k=memory_top_k,
            memory_token_budget=memory_token_budget,
            run_judge=run_judge,
            judge_llm_backend=judge_backend,
            judge_model=judge_model,
            oracle_routing=oracle_routing,
            tool_evolution_enabled=tool_evolution,
            tool_library_id=tool_library or "default",
            tool_evolution_mode=evolution_mode,
        )
        return

    if problem is not None:
        raise typer.BadParameter(
            "--problem without SCENARIO is invalid; pass SCENARIO or use batch mode with --csv."
        )

    benchmark_path = str(csv) if csv is not None else default_benchmark_csv_path()
    run_benchmark_from_csv(
        benchmark_file=benchmark_path,
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        max_attempts=max_attempts,
        parallel=parallel,
        memory_mode=memory_mode,
        memory_bank=memory_bank,
        memory_top_k=memory_top_k,
        memory_token_budget=memory_token_budget,
        run_judge=run_judge,
        judge_llm_backend=judge_backend,
        judge_model=judge_model,
        oracle_routing=oracle_routing,
        tool_evolution_enabled=tool_evolution,
        tool_library_id=tool_library,
        tool_evolution_mode=evolution_mode,
    )
