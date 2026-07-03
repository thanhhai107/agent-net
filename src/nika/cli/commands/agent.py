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
                ),
                memory=MemoryConfig(
                    mode=memory_mode,
                    bank=memory_bank,
                    top_k=memory_k,
                    token_budget=memory_tokens,
                    skill_selector_mode=memory_selector,
                    meta_controller_mode=memory_meta_controller,
                ),
            ),
            session_id=session_id,
            requested_model=model,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
