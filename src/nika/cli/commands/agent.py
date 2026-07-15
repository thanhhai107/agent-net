"""Commands for running diagnosis agents."""

import typer

from agent.composition import AgentRunConfig
from agent.extensions.run import start_agent as start_extension_agent
from nika.utils.agent_config import (
    ENV_AGENT_TYPE,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    ENV_MODEL,
    SUPPORTED_AGENT_TYPES,
    resolve_agent_model,
    resolve_agent_type,
    resolve_llm_provider,
    resolve_max_steps,
)

SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "deepseek", "custom")

agent_app = typer.Typer(help="Troubleshooting agents.")


@agent_app.command("list")
def agent_list() -> None:
    """Print supported agent types and LLM providers."""
    typer.echo("agent_types:")
    for agent_type in SUPPORTED_AGENT_TYPES:
        typer.echo(f"  {agent_type}")
    typer.echo("llm_providers (react, plan-execute, reflexion):")
    for provider in SUPPORTED_LLM_PROVIDERS:
        typer.echo(f"  {provider}")


@agent_app.command("run")
def agent_run(
    agent_type: str | None = typer.Option(
        None,
        "-a",
        "--agent",
        envvar=ENV_AGENT_TYPE,
        help="Workflow: react, plan-execute, or reflexion (required unless NIKA_AGENT_TYPE is in .env).",
    ),
    llm_provider: str | None = typer.Option(
        None,
        "-p",
        "--provider",
        envvar=ENV_LLM_PROVIDER,
        help="LLM provider: openai, ollama, deepseek, custom.",
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
        help="Max steps per phase (required unless NIKA_MAX_STEPS is in .env).",
    ),
    session_id: str | None = typer.Option(
        None, "--session_id", help="Target session id."
    ),
) -> None:
    """Run the agent on the current session task."""
    try:
        resolved_agent_type = resolve_agent_type(agent_type)
        if resolved_agent_type in {"plan-execute", "reflexion"}:
            start_extension_agent(
                AgentRunConfig(
                    agent_type=resolved_agent_type,
                    llm_provider=resolve_llm_provider(
                        llm_provider, agent_type=resolved_agent_type
                    )
                    or "",
                    model=resolve_agent_model(resolved_agent_type, model),
                    max_steps=resolve_max_steps(max_steps),
                ),
                session_id=session_id,
            )
        else:
            from nika.workflows.agent.run import start_agent

            start_agent(
                resolved_agent_type,
                llm_provider,
                model,
                max_steps,
                session_id=session_id,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
