"""Build whitelisted environment variables for sandbox containers."""

from __future__ import annotations

import os
from pathlib import Path

from agent.sandbox.config import (
    ENV_SANDBOX_EXECUTION,
    ENV_SESSION_DIR,
    load_sandbox_env_values,
    sandbox_local_env_file,
)
from agent.sandbox.redact import redact_env_value
from nika.service.mcp_gateway.lifecycle import ENV_GATEWAY_AGENT_URL

# Keys copied from the env file / host environment into the sandbox container.
_SANDBOX_ENV_ALLOWLIST = (
    "NIKA_SESSION_ID",
    "NIKA_SESSION_BACKEND",
    "NIKA_AGENT_TYPE",
    "NIKA_MAX_STEPS",
    "NIKA_MODEL",
    "NIKA_CODEX_MODEL",
    "NIKA_CODEX_SDK_MODEL",
    "NIKA_CODEX_REASONING_EFFORT",
    "NIKA_CLAUDE_SDK_MODEL",
    "NIKA_ENABLE_SKILLS",
    "NIKA_SKILLS_DIR",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
)


def _load_env_file(env_file: Path) -> dict[str, str]:
    return load_sandbox_env_values(env_file, sandbox_local_env_file())


def build_sandbox_env(
    *,
    session_id: str,
    session_dir: str,
    agent_type: str,
    model: str,
    max_steps: int | None,
    reasoning_effort: str | None,
    llm_provider: str | None,
    mcp_gateway_agent_url: str,
    env_file: Path,
    skills_dir: str,
    http_proxy: str | None = None,
    https_proxy: str | None = None,
    no_proxy: str | None = None,
) -> dict[str, str]:
    """Return whitelisted env vars for ``docker run -e`` injection."""
    merged: dict[str, str] = {}
    merged.update(_load_env_file(env_file))
    for key in _SANDBOX_ENV_ALLOWLIST:
        if key in os.environ and os.environ[key].strip():
            merged[key] = os.environ[key].strip()

    merged[ENV_SANDBOX_EXECUTION] = "1"
    merged[ENV_SESSION_DIR] = session_dir
    merged["NIKA_SESSION_ID"] = session_id
    merged["NIKA_AGENT_TYPE"] = agent_type
    merged["NIKA_MODEL"] = model
    session_backend = os.environ.get("NIKA_SESSION_BACKEND", "").strip()
    if session_backend:
        merged["NIKA_SESSION_BACKEND"] = session_backend
    merged[ENV_GATEWAY_AGENT_URL] = mcp_gateway_agent_url.rstrip("/")
    merged["NIKA_SKILLS_DIR"] = skills_dir

    if max_steps is not None:
        merged["NIKA_MAX_STEPS"] = str(max_steps)
    if reasoning_effort:
        merged["NIKA_CODEX_REASONING_EFFORT"] = reasoning_effort
    if llm_provider:
        merged["NIKA_LLM_PROVIDER"] = llm_provider

    if agent_type == "local_cli.codex_cli":
        merged.setdefault("NIKA_CODEX_MODEL", model)
    elif agent_type == "sdk.codex_sdk":
        merged.setdefault("NIKA_CODEX_SDK_MODEL", model)
    elif agent_type in ("local_cli.claude_cli", "sdk.claude_sdk"):
        merged.setdefault("ANTHROPIC_MODEL", model)

    if http_proxy:
        merged["HTTP_PROXY"] = http_proxy
        merged["http_proxy"] = http_proxy
    if https_proxy:
        merged["HTTPS_PROXY"] = https_proxy
        merged["https_proxy"] = https_proxy
    if no_proxy:
        merged["NO_PROXY"] = no_proxy
        merged["no_proxy"] = no_proxy

    return {k: v for k, v in merged.items() if v}


def format_env_for_log(env: dict[str, str]) -> dict[str, str]:
    return {k: redact_env_value(k, v) for k, v in env.items()}
