"""Benchmark runner: full pipeline from env to evaluation."""

from pathlib import Path

import typer

from agent.composition import (
    MemoryConfig,
    ToolEvolutionConfig,
)
from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL
from nika.net_env.net_env_pool import scenario_requires_topo_tier
from nika.utils.agent_config import resolve_max_steps
from nika.workflows.benchmark.inject_defaults import resolve_inject_params
from nika.workflows.benchmark.run import (
    _new_benchmark_results_root,
    default_benchmark_yaml_path,
    is_no_fault_problem,
    run_benchmark_from_yaml,
    run_single_benchmark,
)

benchmark_app = typer.Typer(
    help="Run curated benchmark cases (env → fault → agent → eval)."
)


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
    file: Path | None = typer.Option(
        None,
        "-f",
        "--file",
        help="Benchmark YAML path for batch mode. Defaults to benchmark/benchmark_test.yaml.",
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
    sets: list[str] | None = typer.Option(
        None,
        "--set",
        help="Override inject parameters as key=value (single-case mode).",
    ),
    agent_type: str = typer.Option(
        "react", "-a", "--agent", help="Agent implementation."
    ),
    llm_backend: str = typer.Option(
        DEFAULT_LLM_BACKEND,
        "-b",
        "--backend",
        help="LLM provider (openai, ollama, deepseek, custom).",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "-m", "--model", help="Model id for the agent."
    ),
    max_steps: int | None = typer.Option(
        None,
        "-n",
        "--max-steps",
        help=(
            "Per-worker step limit for LangGraph agents; also the maximum "
            "executed plan items for plan-execute. Defaults to NIKA_MAX_STEPS."
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
    memory_selector: str = typer.Option(
        "lcb",
        "--memory-selector",
        help="Skill-Pro selector: lcb or llm_topk_lcb.",
    ),
    memory_meta_controller: str = typer.Option(
        "heuristic",
        "--memory-meta-controller",
        help="Skill-Pro option termination controller: heuristic or llm.",
    ),
    memory_max_skill_age: int = typer.Option(
        4,
        "--memory-max-skill-age",
        min=1,
        help="Maximum tool transitions controlled by one active Skill-Pro option.",
    ),
    memory_selector_min_lcb: float = typer.Option(
        -0.05,
        "--memory-selector-min-lcb",
        help="Minimum LCB score accepted for mature Skill-Pro options.",
    ),
    memory_selector_nominee_k: int = typer.Option(
        3,
        "--memory-selector-nominee-k",
        min=1,
        help="Number of LLM-nominated skills before LCB selection.",
    ),
    memory_pool_size: int = typer.Option(
        32,
        "--memory-pool-size",
        min=1,
        help="Maximum active Skill-Pro skill pool size.",
    ),
    memory_evolution_threshold: int = typer.Option(
        3,
        "--memory-evolution-threshold",
        min=1,
        help="Minimum replay samples before Skill-Pro refinement/retirement decisions.",
    ),
    memory_best_of_n: int = typer.Option(
        3,
        "--memory-best-of-n",
        min=1,
        help="Number of candidate Skill-Pro procedures proposed per episode.",
    ),
    memory_ppo_epsilon: float = typer.Option(
        0.2,
        "--memory-ppo-epsilon",
        min=0.0,
        help="PPO-style clipping epsilon for Skill-Pro evolution gate.",
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
    tools: str | None = typer.Option(
        None,
        "--tools",
        help="Enable DRAFT Tool Evolution with this documentation library id.",
    ),
    tool_doc_chars: int = typer.Option(
        500,
        "--tool-doc-chars",
        min=100,
        help="Maximum DRAFT refined-doc characters appended to each tool.",
    ),
    tool_prompt_doc_limit: int = typer.Option(
        6,
        "--tool-prompt-doc-limit",
        min=1,
        help="Maximum DRAFT tool docs included in the diagnosis prompt.",
    ),
    tool_scoped_prompt_doc_limit: int = typer.Option(
        4,
        "--tool-scoped-prompt-doc-limit",
        min=1,
        help="Maximum DRAFT docs included when scoped to active tools.",
    ),
    tool_planned_checks: int = typer.Option(
        4,
        "--tool-planned-checks",
        min=0,
        help="Maximum planned DRAFT Explorer checks injected into prompt context.",
    ),
    tool_next_checks: int = typer.Option(
        2,
        "--tool-next-checks",
        min=0,
        help="Maximum DRAFT next-check suggestions shown per tool.",
    ),
    tool_convergence_threshold: float = typer.Option(
        0.75,
        "--tool-convergence-threshold",
        min=0.0,
        max=1.0,
        help="DRAFT documentation convergence threshold for freezing docs.",
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
        help="Internal zero-based YAML case index for timeline reporting.",
    ),
) -> None:
    """Run one benchmark case from YAML, or a single case when SCENARIO and --problem are set."""
    if not run_judge and (judge_backend is not None or judge_model is not None):
        raise typer.BadParameter(
            "Pass --judge to enable LLM judge; omit --judge-backend/--judge-model otherwise."
        )
    judge_backend = judge_backend or DEFAULT_LLM_BACKEND
    judge_model = judge_model or DEFAULT_MODEL
    if memory is not None and memory_read is not None:
        raise typer.BadParameter("Use either --memory or --memory-read, not both.")
    if memory_selector not in {"lcb", "llm_topk_lcb"}:
        raise typer.BadParameter("--memory-selector must be lcb or llm_topk_lcb.")
    if memory_meta_controller not in {"heuristic", "llm"}:
        raise typer.BadParameter(
            "--memory-meta-controller must be heuristic or llm."
        )
    memory_mode = "evolve" if memory is not None else "read" if memory_read else "off"
    memory_bank = memory or memory_read or "default"
    resolved_max_steps = resolve_max_steps(max_steps)
    tool_evolution_enabled = tools is not None
    tool_library_id = tools or "default"
    memory_config = MemoryConfig(
        mode=memory_mode,
        bank=memory_bank,
        top_k=memory_k,
        token_budget=memory_tokens,
        skill_selector_mode=memory_selector,
        meta_controller_mode=memory_meta_controller,
        max_skill_age=memory_max_skill_age,
        selector_min_lcb=memory_selector_min_lcb,
        selector_nominee_k=memory_selector_nominee_k,
        pool_size=memory_pool_size,
        evolution_threshold=memory_evolution_threshold,
        best_of_n=memory_best_of_n,
        ppo_epsilon=memory_ppo_epsilon,
    )
    tool_config = ToolEvolutionConfig(
        enabled=tool_evolution_enabled,
        library_id=tool_library_id,
        tool_doc_chars=tool_doc_chars,
        prompt_doc_limit=tool_prompt_doc_limit,
        scoped_prompt_doc_limit=tool_scoped_prompt_doc_limit,
        planned_checks=tool_planned_checks,
        next_checks=tool_next_checks,
        convergence_threshold=tool_convergence_threshold,
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
        if is_no_fault_problem(problem):
            if sets:
                raise typer.BadParameter(
                    "--set is invalid for no-fault benchmark cases."
                )
            inject_params = {}
        else:
            inject_params = resolve_inject_params(problem, scenario, topo)
            inject_params.update(_parse_set_options(sets))
        run_single_benchmark(
            problem=problem,
            scenario=scenario,
            topo_size=topo,
            agent_type=agent_type,
            llm_backend=llm_backend,
            model=model,
            max_steps=resolved_max_steps,
            max_attempts=max_attempts,
            memory=memory_config,
            run_judge=run_judge,
            judge_llm_backend=judge_backend,
            judge_model=judge_model,
            tool_evolution=tool_config,
            result_root=benchmark_root,
            fault_seed=fault_seed,
            benchmark_index=benchmark_index,
            inject_params=inject_params,
        )
        return

    if problem is not None:
        raise typer.BadParameter(
            "--problem without SCENARIO is invalid; pass SCENARIO or use batch mode with --file."
        )
    if sets:
        raise typer.BadParameter("--set applies to single-case mode only.")
    benchmark_path = str(file) if file is not None else default_benchmark_yaml_path()
    run_benchmark_from_yaml(
        benchmark_file=benchmark_path,
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=resolved_max_steps,
        max_attempts=max_attempts,
        memory=memory_config,
        run_judge=run_judge,
        judge_llm_backend=judge_backend,
        judge_model=judge_model,
        tool_evolution=tool_config,
        result_root=result_root,
    )
