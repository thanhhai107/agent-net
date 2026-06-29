"""Commands for SIA-style agent evolution runs."""

from pathlib import Path

import typer

from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from agent.tool_evolution.models import ToolEvolutionMode
from nika.workflows.benchmark.run import default_benchmark_csv_path
from nika.workflows.evolve.run import FEEDBACK_MODES, run_agent_evolution

evolve_app = typer.Typer(
    help="Run an outer-loop agent evolution experiment over benchmark generations."
)


@evolve_app.command("run")
def evolve_run(
    file: Path = typer.Option(
        Path(default_benchmark_csv_path()),
        "-f",
        "--file",
        help="Benchmark CSV path. Defaults to benchmark/benchmark_test.csv.",
    ),
    max_gen: int = typer.Option(
        3,
        "--max-gen",
        min=1,
        help="Number of outer-loop generations.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Stable id for this evolution run. Defaults to a timestamp id.",
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
        help="Per-worker step limit for LangGraph agents.",
    ),
    max_attempts: int = typer.Option(
        3,
        "-r",
        "--max-attempts",
        min=1,
        help="Maximum attempts for reflexion; ignored by other agents.",
    ),
    parallel: int = typer.Option(
        1,
        "-j",
        "--parallel",
        min=1,
        help="Number of benchmark cases to run concurrently within each generation.",
    ),
    tools: str | None = typer.Option(
        None,
        "--tools",
        help="Also enable Tool Evolution with this library id.",
    ),
    tool_mode: str = typer.Option(
        ToolEvolutionMode.DUAL.value,
        "--tool-mode",
        help="Tool Evolution mode: mastery, distill, dual.",
    ),
    initial_policy: Path | None = typer.Option(
        None,
        "--initial-policy",
        help="Optional policy overlay injected into generation 1.",
    ),
    feedback_mode: str = typer.Option(
        "auto",
        "--feedback-mode",
        help="Next-policy planner: auto, deterministic, or llm.",
    ),
    feedback_backend: str | None = typer.Option(
        None,
        "--feedback-backend",
        help="LLM provider for the feedback agent. Defaults to --backend.",
    ),
    feedback_model: str | None = typer.Option(
        None,
        "--feedback-model",
        help="Model id for the feedback agent. Defaults to --model.",
    ),
    run_judge: bool = typer.Option(
        False,
        "--judge",
        help="Run LLM-as-judge after metrics in each generation.",
    ),
    judge_backend: str | None = typer.Option(
        None,
        "--judge-backend",
        help="LLM provider for the judge when --judge is set.",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Model id for the judge when --judge is set.",
    ),
    oracle_routing: bool = typer.Option(
        False,
        "--oracle-routing",
        help="Use hidden problem labels for MCP server selection (oracle baseline).",
    ),
    runtime_root: Path | None = typer.Option(
        None,
        "--runtime-root",
        hidden=True,
        help="Internal test hook for runtime artifact root.",
    ),
    results_root: Path | None = typer.Option(
        None,
        "--results-root",
        hidden=True,
        help="Internal test hook for benchmark result root.",
    ),
) -> None:
    """Run benchmark generations and feed scored artifacts into the next policy overlay."""
    if not run_judge and (judge_backend is not None or judge_model is not None):
        raise typer.BadParameter(
            "Pass --judge to enable LLM judge; omit --judge-backend/--judge-model otherwise."
        )
    try:
        ToolEvolutionMode(tool_mode)
    except ValueError as exc:
        raise typer.BadParameter(
            "tool_mode must be one of "
            + ", ".join(item.value for item in ToolEvolutionMode)
        ) from exc
    if feedback_mode not in FEEDBACK_MODES:
        raise typer.BadParameter(
            "feedback_mode must be one of " + ", ".join(sorted(FEEDBACK_MODES))
        )

    kwargs = {}
    if runtime_root is not None:
        kwargs["runtime_root"] = runtime_root
    if results_root is not None:
        kwargs["results_root"] = results_root

    run_agent_evolution(
        benchmark_file=file,
        max_generations=max_gen,
        run_id=run_id,
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        max_attempts=max_attempts,
        parallel=parallel,
        run_judge=run_judge,
        judge_llm_backend=judge_backend or DEFAULT_LLM_BACKEND,
        judge_model=judge_model or DEFAULT_MODEL,
        oracle_routing=oracle_routing,
        tool_evolution_enabled=tools is not None,
        tool_library_id=tools,
        tool_evolution_mode=tool_mode,
        initial_policy_overlay=initial_policy,
        feedback_mode=feedback_mode,
        feedback_llm_backend=feedback_backend or llm_backend,
        feedback_model=feedback_model or model,
        **kwargs,
    )
