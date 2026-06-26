"""Benchmark runner: full pipeline from env to evaluation."""

from pathlib import Path

import typer

from nika.net_env.net_env_pool import scenario_requires_topo_size
from nika.workflows.benchmark.run import (
    default_benchmark_csv_path,
    run_benchmark_from_csv,
    run_single_benchmark,
)

benchmark_app = typer.Typer(help="Run curated benchmark cases (env → fault → agent → eval).")


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
    size: str | None = typer.Option(
        None,
        "-s",
        "--size",
        help="Topology size s, m, or l (required only for scalable scenarios).",
    ),
    agent_type: str = typer.Option("react", "-a", "--agent", help="Agent implementation."),
    llm_provider: str = typer.Option("openai", "-p", "--provider", help="LLM provider (openai, ollama, deepseek)."),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model id for the agent (claude: defaults from ANTHROPIC_MODEL in .env).",
    ),
    max_steps: int = typer.Option(
        20,
        "-n",
        "--max-steps",
        help="Max ReAct steps (react and mock only; ignored for cli).",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help=(
            "CSV batch mode: number of rows to run simultaneously per batch. "
            "Rows are chunked into groups of this size; each group runs fully in "
            "parallel before the next group starts (default: 1)."
        ),
    ),
    run_judge: bool = typer.Option(
        False,
        "--judge",
        help="Run LLM-as-judge after metrics (default: metrics only).",
    ),
    judge_provider: str | None = typer.Option(
        None,
        "--judge-provider",
        help="LLM provider for the judge (required with --judge).",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        help="Model id for the judge (required with --judge).",
    ),
) -> None:
    """Run one benchmark row from CSV, or a single case when SCENARIO and --problem are set."""
    if run_judge:
        if not judge_provider or not judge_model:
            raise typer.BadParameter("--judge-provider and --judge-model are required when --judge is set.")
    elif judge_provider is not None or judge_model is not None:
        raise typer.BadParameter("Pass --judge to enable LLM judge; omit --judge-provider/--judge-model otherwise.")

    if scenario is not None and csv is not None:
        raise typer.BadParameter("Use either SCENARIO (single-case mode) or --csv (batch mode), not both.")

    single_mode = scenario is not None

    if single_mode:
        if batch_size != 1:
            raise typer.BadParameter("--batch-size applies to CSV batch mode only; omit it for a single case.")
        if not problem:
            raise typer.BadParameter("--problem is required when SCENARIO is given.")
        if scenario_requires_topo_size(scenario) and not size:
            raise typer.BadParameter(f"Scenario '{scenario}' requires -s/--size (s, m, or l).")
        if not scenario_requires_topo_size(scenario) and size is not None:
            raise typer.BadParameter(f"Scenario '{scenario}' does not use sizes; omit -s/--size.")
        topo = size or ""
        run_single_benchmark(
            problem=problem,
            scenario=scenario,
            topo_size=topo,
            agent_type=agent_type,
            llm_provider=llm_provider,
            model=model,
            max_steps=max_steps,
            run_judge=run_judge,
            judge_llm_provider=judge_provider,
            judge_model=judge_model,
        )
        return

    if problem is not None:
        raise typer.BadParameter("--problem without SCENARIO is invalid; pass SCENARIO or use batch mode with --csv.")

    benchmark_path = str(csv) if csv is not None else default_benchmark_csv_path()
    run_benchmark_from_csv(
        benchmark_file=benchmark_path,
        agent_type=agent_type,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        batch_size=batch_size,
        run_judge=run_judge,
        judge_llm_provider=judge_provider,
        judge_model=judge_model,
    )
