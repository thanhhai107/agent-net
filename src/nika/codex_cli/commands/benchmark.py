"""Benchmark runner: full pipeline from env to evaluation."""

from pathlib import Path

import typer

from agent.composition import (
    HarnessConfig,
    MemoryConfig,
    ToolEvolutionConfig,
)
from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.tool_evolution.models import ToolEvolutionMode
from nika.net_env.net_env_pool import scenario_requires_topo_tier
from nika.workflows.benchmark.run import (
    _new_benchmark_results_root,
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
    file: Path | None = typer.Option(
        None,
        "-f",
        "--file",
        help="Benchmark CSV path for batch mode. Defaults to benchmark/benchmark_test.csv.",
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
        DEFAULT_MAX_STEPS,
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
    memory: str | None = typer.Option(
        None,
        "--memory",
        help="Enable evolving memory with this bank id.",
    ),
    memory_read: str | None = typer.Option(
        None,
        "--memory-read",
        help="Read this memory bank without updating it.",
    ),
    memory_k: int = typer.Option(
        5,
        "--memory-k",
        min=1,
        max=20,
        help="Maximum memories injected into one diagnosis.",
    ),
    memory_tokens: int = typer.Option(
        1500,
        "--memory-tokens",
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
    tools: str | None = typer.Option(
        None,
        "--tools",
        help="Enable Tool Evolution with this library id.",
    ),
    tool_mode: str = typer.Option(
        ToolEvolutionMode.DUAL.value,
        "--tool-mode",
        help="Tool Evolution mode: mastery, distill, dual.",
    ),
    result_root: Path | None = typer.Option(
        None,
        "--result-root",
        hidden=True,
        help="Internal benchmark result root for row subprocess execution.",
    ),
    fault_seed: str | None = typer.Option(
        None,
        "--fault-seed",
        hidden=True,
        help="Internal deterministic fault seed for row subprocess execution.",
    ),
    benchmark_index: int | None = typer.Option(
        None,
        "--benchmark-index",
        hidden=True,
        help="Internal zero-based CSV row index for timeline reporting.",
    ),
    harness: Path | None = typer.Option(
        None,
        "--harness",
        hidden=True,
        help="Internal SIA-H target_agent.py path.",
    ),
    harness_allow_failure: bool = typer.Option(
        False,
        "--harness-allow-failure",
        hidden=True,
        help="Internal flag for evolution runs that should score crashed targets.",
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
        ToolEvolutionMode(tool_mode)
    except ValueError as exc:
        raise typer.BadParameter(
            "tool_mode must be one of "
            + ", ".join(item.value for item in ToolEvolutionMode)
        ) from exc
    if memory is not None and memory_read is not None:
        raise typer.BadParameter("Use either --memory or --memory-read, not both.")
    memory_mode = "evolve" if memory is not None else "read" if memory_read else "off"
    memory_bank = memory or memory_read or "default"
    tool_evolution_enabled = tools is not None
    tool_library_id = tools or "default"
    memory_config = MemoryConfig(
        mode=memory_mode,
        bank=memory_bank,
        top_k=memory_k,
        token_budget=memory_tokens,
    )
    tool_config = ToolEvolutionConfig(
        enabled=tool_evolution_enabled,
        library_id=tool_library_id,
        mode=tool_mode,
    )
    harness_config = HarnessConfig(
        target_agent_path=str(harness) if harness else None
    )

    if scenario is not None and file is not None:
        raise typer.BadParameter(
            "Use either SCENARIO (single-case mode) or --file (batch mode), not both."
        )

    single_mode = scenario is not None

    if single_mode:
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
        benchmark_root = result_root or _new_benchmark_results_root(scenario)
        run_single_benchmark(
            problem=problem,
            scenario=scenario,
            topo_size=topo,
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            memory=memory_config,
            run_judge=run_judge,
            judge_llm_backend=judge_backend,
            judge_model=judge_model,
            oracle_routing=oracle_routing,
            tool_evolution=tool_config,
            harness=harness_config,
            harness_allow_failure=harness_allow_failure,
            result_root=benchmark_root,
            fault_seed=fault_seed,
            benchmark_index=benchmark_index,
        )
        return

    if problem is not None:
        raise typer.BadParameter(
            "--problem without SCENARIO is invalid; pass SCENARIO or use batch mode with --file."
        )
    benchmark_path = str(file) if file is not None else default_benchmark_csv_path()
    run_benchmark_from_csv(
        benchmark_file=benchmark_path,
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        max_attempts=max_attempts,
        memory=memory_config,
        run_judge=run_judge,
        judge_llm_backend=judge_backend,
        judge_model=judge_model,
        oracle_routing=oracle_routing,
        tool_evolution=tool_config,
        harness=harness_config,
        harness_allow_failure=harness_allow_failure,
        result_root=result_root,
    )
