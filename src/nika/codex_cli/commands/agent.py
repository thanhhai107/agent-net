"""Commands for running diagnosis agents."""

import typer

from agent.cli.codex_worker import REASONING_EFFORT_LEVELS
from agent.llm.model_factory import NETMIND_SUPPORTED_MODELS

SUPPORTED_AGENT_TYPES = ("react", "plan-execute", "reflexion", "mock", "cli")
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
    typer.echo("reasoning_effort (cli only):")
    for level in REASONING_EFFORT_LEVELS:
        typer.echo(f"  {level}")


@agent_app.command("run")
def agent_run(
    agent_type: str = typer.Option("react", "-a", "--agent", help="Agent implementation."),
    llm_backend: str = typer.Option(
        "openai",
        "-b",
        "--backend",
        help="LLM provider (openai, ollama, deepseek, netmind).",
    ),
    model: str = typer.Option("gpt-5-mini", "-m", "--model", help="Model id for the chosen backend."),
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
    reasoning_effort: str | None = typer.Option(
        None,
        "-e",
        "--reasoning-effort",
        help="Codex model_reasoning_effort (cli only): none, minimal, low, medium, high, xhigh.",
    ),
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
) -> None:
    """Run the agent on the current session task."""
    from nika.workflows.agent.run import start_agent

    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise typer.BadParameter(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORT_LEVELS)}"
        )

    try:
        start_agent(
            agent_type,
            llm_backend,
            model,
            max_steps,
            max_attempts=max_attempts,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
