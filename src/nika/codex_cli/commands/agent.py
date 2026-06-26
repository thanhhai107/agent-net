"""Commands for running diagnosis agents."""

import typer

from agent.codex_cli.codex_worker import REASONING_EFFORT_LEVELS
from agent.composition import (
    AgentRunConfig,
    MemoryConfig,
    ToolEvolutionConfig,
)
from agent.defaults import DEFAULT_MAX_STEPS
from agent.llm.model_factory import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_MODEL,
    NETMIND_SUPPORTED_MODELS,
)

SUPPORTED_AGENT_TYPES = (
    "react",
    "plan-execute",
    "reflexion",
    "mock",
    "cli",
    "codex_cli",
    "claude_cli",
)
SUPPORTED_LLM_BACKENDS = ("openai", "ollama", "deepseek", "netmind")

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
    typer.echo("netmind_models:")
    for model in NETMIND_SUPPORTED_MODELS:
        typer.echo(f"  {model}")
    typer.echo("reasoning_effort (cli/codex_cli only):")
    for level in REASONING_EFFORT_LEVELS:
        typer.echo(f"  {level}")


@agent_app.command("run")
def agent_run(
    agent_type: str = typer.Option(
        "react", "-a", "--agent", help="Agent implementation."
    ),
    llm_backend: str = typer.Option(
        DEFAULT_LLM_BACKEND,
        "-b",
        "--backend",
        help="LLM provider (openai, ollama, deepseek, netmind).",
    ),
    model: str | None = typer.Option(
        None, "-m", "--model", help="Model id for the chosen backend or CLI agent."
    ),
    max_steps: int = typer.Option(
        DEFAULT_MAX_STEPS,
        "-n",
        "--max-steps",
        help=(
            "Per-worker step limit for LangGraph agents; also the maximum "
            "executed plan items for plan-execute. Ignored for cli/codex_cli/claude_cli."
        ),
    ),
    max_attempts: int = typer.Option(
        3,
        "-r",
        "--max-attempts",
        min=1,
        help="Maximum attempts for the reflexion agent; ignored by other agents.",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "-e",
        "--reasoning-effort",
        help="Codex model_reasoning_effort (cli/codex_cli only): none, minimal, low, medium, high, xhigh.",
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
) -> None:
    """Run the agent on the current session task."""
    from nika.workflows.agent.run import start_agent

    normalized_agent = agent_type.lower()
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise typer.BadParameter(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORT_LEVELS)}"
        )
    if reasoning_effort is not None and normalized_agent not in {"cli", "codex_cli"}:
        raise typer.BadParameter("--reasoning-effort is supported only for cli/codex_cli.")
    if memory is not None and memory_read is not None:
        raise typer.BadParameter("Use either --memory or --memory-read, not both.")
    memory_mode = "evolve" if memory is not None else "read" if memory_read else "off"
    memory_bank = memory or memory_read or "default"

    try:
        start_agent(
            AgentRunConfig(
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model or DEFAULT_MODEL,
                max_steps=max_steps,
                max_attempts=max_attempts,
                reasoning_effort=reasoning_effort,
                tool_evolution=ToolEvolutionConfig(
                    enabled=tools is not None,
                    library_id=tools or "default",
                ),
                memory=MemoryConfig(
                    mode=memory_mode,
                    bank=memory_bank,
                    top_k=memory_k,
                    token_budget=memory_tokens,
                ),
            ),
            session_id=session_id,
            requested_model=model,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
