"""Commands for running diagnosis agents."""

import typer

from agent.codex_cli.codex_worker import REASONING_EFFORT_LEVELS

SUPPORTED_AGENT_TYPES = ("react", "mock", "codex_cli", "claude_cli")
SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "deepseek")

agent_app = typer.Typer(help="Troubleshooting agents.")


@agent_app.command("list")
def agent_list() -> None:
    """Print supported agent types and LLM providers."""
    typer.echo("agent_types:")
    for agent_type in SUPPORTED_AGENT_TYPES:
        typer.echo(f"  {agent_type}")
    typer.echo("llm_providers:")
    for provider in SUPPORTED_LLM_PROVIDERS:
        typer.echo(f"  {provider}")
    typer.echo("reasoning_effort (codex_cli agent only):")
    for level in REASONING_EFFORT_LEVELS:
        typer.echo(f"  {level}")


@agent_app.command("run")
def agent_run(
    agent_type: str = typer.Option("react", "-a", "--agent", help="Agent implementation."),
    llm_provider: str = typer.Option("openai", "-p", "--provider", help="LLM provider (openai, ollama, deepseek)."),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model id for the chosen provider or agent (claude: defaults from ANTHROPIC_MODEL in .env).",
    ),
    max_steps: int = typer.Option(
        20,
        "-n",
        "--max-steps",
        help="Max ReAct steps (react and mock only; ignored for cli).",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "-e",
        "--reasoning-effort",
        help="Codex model_reasoning_effort (cli agent only): none, minimal, low, medium, high, xhigh.",
    ),
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id."),
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
            llm_provider,
            model,
            max_steps,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
