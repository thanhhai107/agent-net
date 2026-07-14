"""Commands for running diagnosis agents."""

import typer

from agent.composition import AgentRunConfig
from agent.extensions.run import start_agent as start_extension_agent
from agent.sandbox.config import (
    ENV_AGENT_SANDBOX,
    ENV_SANDBOX_CPUS,
    ENV_SANDBOX_ENV_FILE,
    ENV_SANDBOX_IMAGE,
    ENV_SANDBOX_KEEP,
    ENV_SANDBOX_MEMORY,
    ENV_SANDBOX_NETWORK,
)
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
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        envvar=ENV_AGENT_SANDBOX,
        help="Run the agent inside the Docker sandbox container.",
    ),
    sandbox_image: str | None = typer.Option(
        None,
        "--sandbox-image",
        envvar=ENV_SANDBOX_IMAGE,
        help="Docker image for sandbox execution (default: nika/agent-sandbox:latest).",
    ),
    sandbox_env_file: str | None = typer.Option(
        None,
        "--sandbox-env-file",
        envvar=ENV_SANDBOX_ENV_FILE,
        help="Env file for whitelisted credential injection into the sandbox.",
    ),
    sandbox_keep_container: bool = typer.Option(
        False,
        "--sandbox-keep-container",
        envvar=ENV_SANDBOX_KEEP,
        help="Do not remove the sandbox container after the agent exits.",
    ),
    sandbox_cpus: str | None = typer.Option(
        None,
        "--sandbox-cpus",
        envvar=ENV_SANDBOX_CPUS,
        help="CPU limit for the sandbox container (docker --cpus).",
    ),
    sandbox_memory: str | None = typer.Option(
        None,
        "--sandbox-memory",
        envvar=ENV_SANDBOX_MEMORY,
        help="Memory limit for the sandbox container (docker --memory).",
    ),
    sandbox_network: str | None = typer.Option(
        None,
        "--sandbox-network",
        envvar=ENV_SANDBOX_NETWORK,
        help="Docker network mode for sandbox (bridge or host; use host with Clash TUN).",
    ),
) -> None:
    """Run the agent on the current session task."""
    try:
        resolved_agent_type = resolve_agent_type(agent_type)
        if resolved_agent_type in {"plan-execute", "reflexion"}:
            if (
                sandbox
                or sandbox_keep_container
                or any(
                    value is not None
                    for value in (
                        sandbox_image,
                        sandbox_env_file,
                        sandbox_cpus,
                        sandbox_memory,
                        sandbox_network,
                    )
                )
            ):
                raise ValueError(
                    "Sandbox execution is currently supported only for react."
                )
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
                sandbox=sandbox,
                sandbox_image=sandbox_image,
                sandbox_env_file=sandbox_env_file,
                sandbox_keep_container=sandbox_keep_container,
                sandbox_cpus=sandbox_cpus,
                sandbox_memory=sandbox_memory,
                sandbox_network=sandbox_network,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
