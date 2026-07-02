"""Benchmark runner: full pipeline from env to evaluation."""

from pathlib import Path

import typer

from nika.net_env.net_env_pool import scenario_requires_topo_size
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_JUDGE_MODEL,
    ENV_JUDGE_PROVIDER,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    ENV_MODEL,
)
from nika.workflows.benchmark.run import (
    default_benchmark_yaml_path,
    run_benchmark_from_yaml,
    run_single_case,
    validate_inject_params,
)

benchmark_app = typer.Typer(help="Run curated benchmark cases (env → fault → agent → eval).")


def _parse_set_options(raw_items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_items or []:
        if "=" not in raw:
            raise typer.BadParameter(f"Invalid --set value {raw!r}. Use key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Invalid --set value {raw!r}. Key cannot be empty.")
        overrides[key] = value.strip()
    return overrides


@benchmark_app.command("run")
def benchmark_run(
    scenario: str | None = typer.Argument(
        default=None,
        metavar="SCENARIO",
        help="Scenario id for a single case (omit for YAML batch mode).",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Benchmark YAML path (batch mode). Defaults to benchmark/benchmark_selected.yaml under the repo root.",
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
    sets: list[str] | None = typer.Option(
        None,
        "--set",
        help="Inject parameters as key=value (required in single-case mode).",
    ),
    agent_type: str | None = typer.Option(
        None,
        "-a",
        "--agent",
        envvar=ENV_AGENT_TYPE,
        help="Agent implementation (required unless NIKA_AGENT_TYPE is in .env).",
    ),
    llm_provider: str | None = typer.Option(
        None,
        "-p",
        "--provider",
        envvar=ENV_LLM_PROVIDER,
        help="LLM provider for byo.langgraph only: openai, ollama, deepseek.",
    ),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        envvar=ENV_MODEL,
        help="Model id (required unless agent-specific NIKA_*_MODEL or NIKA_MODEL is in .env).",
    ),
    max_steps: int | None = typer.Option(
        None,
        "-n",
        "--max-steps",
        envvar=ENV_MAX_STEPS,
        help="Max steps per phase (required unless NIKA_MAX_STEPS is in .env; byo.langgraph, byo.mcp_agent, byo.autogen, community.sade).",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help=(
            "YAML batch mode: number of rows to run simultaneously per batch. "
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
        envvar=ENV_JUDGE_PROVIDER,
        help="LLM provider for the judge (required with --judge unless set in .env).",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        envvar=ENV_JUDGE_MODEL,
        help="Model id for the judge (required with --judge unless set in .env).",
    ),
) -> None:
    """Run one benchmark row from YAML, or a single case when SCENARIO and --problem are set."""
    if run_judge:
        from nika.utils.agent_config import resolve_judge_model, resolve_judge_provider

        judge_provider = resolve_judge_provider(judge_provider)
        judge_model = resolve_judge_model(judge_model)
    elif judge_provider is not None or judge_model is not None:
        raise typer.BadParameter("Pass --judge to enable LLM judge; omit --judge-provider/--judge-model otherwise.")

    if scenario is not None and config is not None:
        raise typer.BadParameter("Use either SCENARIO (single-case mode) or --config (batch mode), not both.")

    single_mode = scenario is not None

    if single_mode:
        if batch_size != 1:
            raise typer.BadParameter("--batch-size applies to YAML batch mode only; omit it for a single case.")
        if not problem:
            raise typer.BadParameter("--problem is required when SCENARIO is given.")
        if scenario_requires_topo_size(scenario) and not size:
            raise typer.BadParameter(f"Scenario '{scenario}' requires -s/--size (s, m, or l).")
        if not scenario_requires_topo_size(scenario) and size is not None:
            raise typer.BadParameter(f"Scenario '{scenario}' does not use sizes; omit -s/--size.")
        topo = size or ""
        inject_params = _parse_set_options(sets)
        if not inject_params:
            raise typer.BadParameter(
                "Single-case mode requires complete inject parameters via --set key=value. "
                "Use batch mode with --config to run curated YAML cases."
            )
        try:
            validate_inject_params(problem, scenario, topo, inject_params)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        run_single_case(
            problem=problem,
            scenario=scenario,
            topo_size=topo,
            agent_type=agent_type,
            llm_provider=llm_provider,
            model=model,
            max_steps=max_steps,
            inject_params=inject_params,
            run_judge=run_judge,
            judge_llm_provider=judge_provider,
            judge_model=judge_model,
        )
        return

    if problem is not None:
        raise typer.BadParameter("--problem without SCENARIO is invalid; pass SCENARIO or use batch mode with --config.")

    benchmark_path = str(config) if config is not None else default_benchmark_yaml_path()
    run_benchmark_from_yaml(
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
