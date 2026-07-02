"""SADE credential helpers (aligned with ``local_cli.claude_cli``)."""

from __future__ import annotations

from agent.local_cli.claude_cli.config import (
    has_env_claude_credentials,
    prepare_claude_subprocess_env,
)


def sade_credentials_available() -> bool:
    """True when Anthropic-compatible API credentials are configured in env."""
    return has_env_claude_credentials()


def prepare_sade_sdk_env(*, session_id: str) -> dict[str, str]:
    """Build SDK env with DeepSeek/Anthropic credentials and session context."""
    if not has_env_claude_credentials():
        raise RuntimeError(
            "Anthropic-compatible credentials are not set. Configure ANTHROPIC_API_KEY "
            "or ANTHROPIC_AUTH_TOKEN (+ optional ANTHROPIC_BASE_URL) in the repo-root "
            ".env before running `nika agent run -a community.sade`."
        )
    env = prepare_claude_subprocess_env()
    env["NIKA_SESSION_ID"] = session_id
    return env
