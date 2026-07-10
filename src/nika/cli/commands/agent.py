"""Commands for running diagnosis agents."""

import typer

from agent.composition import (
    AgentRunConfig,
    MemoryConfig,
    ToolEvolutionConfig,
)
from agent.llm.model_factory import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
)
from nika.utils.agent_config import resolve_max_steps

SUPPORTED_AGENT_TYPES = (
    "react",
    "plan-execute",
    "reflexion",
    "mock",
)
SUPPORTED_LLM_BACKENDS = ("openai", "ollama", "deepseek", "custom")

agent_app = typer.Typer(help="Troubleshooting agents.")


@agent_app.command("list")
def agent_list() -> None:
    """Print supported agent types and LLM backends."""
    typer.echo("agent_types:")
    for agent_type in SUPPORTED_AGENT_TYPES:
        typer.echo(f"  {agent_type}")
    typer.echo("llm_backends:")
    for backend in SUPPORTED_LLM_BACKENDS:
        typer.echo(f"  {backend}")


@agent_app.command("run")
def agent_run(
    agent_type: str = typer.Option(
        "react", "-a", "--agent", help="Agent implementation."
    ),
    llm_backend: str = typer.Option(
        DEFAULT_LLM_BACKEND,
        "-b",
        "--backend",
        help="LLM provider (openai, ollama, deepseek, custom).",
    ),
    model: str | None = typer.Option(
        None, "-m", "--model", help="Model id for the chosen backend."
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
    session_id: str | None = typer.Option(
        None, "--session-id", help="Target session id (lab_hash)."
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
    memory_expert_seeds: bool = typer.Option(
        False,
        "--memory-expert-seeds",
        help="Enable optional NIKA expert seed ladders; core Skill-Pro uses generic seeds only.",
    ),
) -> None:
    """Run the agent on the current session task."""
    from nika.workflows.agent.run import start_agent

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

    try:
        start_agent(
            AgentRunConfig(
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model or DEFAULT_MODEL,
                max_steps=resolved_max_steps,
                max_attempts=max_attempts,
                tool_evolution=ToolEvolutionConfig(
                    enabled=tools is not None,
                    library_id=tools or "default",
                    tool_doc_chars=tool_doc_chars,
                    prompt_doc_limit=tool_prompt_doc_limit,
                    scoped_prompt_doc_limit=tool_scoped_prompt_doc_limit,
                    planned_checks=tool_planned_checks,
                    next_checks=tool_next_checks,
                    convergence_threshold=tool_convergence_threshold,
                ),
                memory=MemoryConfig(
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
                    include_expert_seeds=memory_expert_seeds,
                ),
            ),
            session_id=session_id,
            requested_model=model,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
