"""Commands for running diagnosis agents."""

import typer

from agent.local_cli.codex_cli.codex_worker import REASONING_EFFORT_LEVELS
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
    ENV_CODEX_REASONING_EFFORT,
    ENV_LLM_PROVIDER,
    ENV_MAX_STEPS,
    ENV_MODEL,
)

SUPPORTED_AGENT_TYPES = (
    "byo.langgraph",
    "byo.mcp_agent",
    "byo.autogen",
    "local_cli.codex_cli",
    "local_cli.claude_cli",
    "community.sade",
    "sdk.claude_sdk",
    "sdk.codex_sdk",
)
SUPPORTED_LLM_PROVIDERS = ("openai", "ollama", "deepseek", "custom")

agent_app = typer.Typer(help="Troubleshooting agents.")


@agent_app.command("list")
def agent_list() -> None:
    """Print supported agent types and LLM providers."""
    typer.echo("agent_types:")
    for agent_type in SUPPORTED_AGENT_TYPES:
        typer.echo(f"  {agent_type}")
    typer.echo("llm_providers (byo.langgraph only):")
    for provider in SUPPORTED_LLM_PROVIDERS:
        typer.echo(f"  {provider}")
    typer.echo("reasoning_effort (local_cli.codex_cli, sdk.codex_sdk):")
    for level in REASONING_EFFORT_LEVELS:
        typer.echo(f"  {level}")


@agent_app.command("run")
def agent_run(
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
        help="LLM provider for byo.langgraph only: openai, ollama, deepseek, custom.",
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
        help="Max steps per phase (required unless NIKA_MAX_STEPS is in .env; byo.langgraph, byo.mcp_agent, byo.autogen, community.sade, sdk.claude_sdk).",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "-e",
        "--reasoning-effort",
        envvar=ENV_CODEX_REASONING_EFFORT,
        help="Codex model_reasoning_effort (local_cli.codex_cli, sdk.codex_sdk): none, minimal, low, medium, high, xhigh.",
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
