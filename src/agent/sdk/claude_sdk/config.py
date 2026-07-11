"""Credential helpers for sdk.claude_sdk (DeepSeek via Anthropic-compatible API)."""

from __future__ import annotations

from agent.local_cli.claude_cli.config import (
    has_env_claude_credentials,
    prepare_claude_subprocess_env,
    resolve_claude_model,
)


def claude_sdk_credentials_available() -> bool:
    """True when Anthropic-compatible API credentials are configured in env."""
    return has_env_claude_credentials()


def prepare_claude_sdk_env(*, session_id: str) -> dict[str, str]:
    """Build SDK env with DeepSeek/Anthropic credentials and session context."""
    if not has_env_claude_credentials():
        raise RuntimeError(
            "Anthropic-compatible credentials are not set. Configure ANTHROPIC_API_KEY "
            "or ANTHROPIC_AUTH_TOKEN (+ optional ANTHROPIC_BASE_URL) in the repo-root "
            ".env before running `nika agent run -a sdk.claude_sdk`."
        )
    env = prepare_claude_subprocess_env()
    env["NIKA_SESSION_ID"] = session_id
    return env


def resolve_claude_sdk_model(model: str | None) -> str:
    """Use *model* when set; otherwise fall back to the Claude CLI model chain."""
    return resolve_claude_model(model)
