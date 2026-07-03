"""Agent CLI configuration from environment variables.

No hard-coded defaults: each setting must come from a CLI flag or ``.env``.
CLI flags take precedence when both are set.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Shared CLI options (nika agent run / nika benchmark run)
ENV_AGENT_TYPE = "NIKA_AGENT_TYPE"
ENV_LLM_PROVIDER = "NIKA_LLM_PROVIDER"
ENV_MAX_STEPS = "NIKA_MAX_STEPS"
ENV_MODEL = "NIKA_MODEL"

# ReAct agent
ENV_REACT_MODEL = "NIKA_REACT_MODEL"

# Mock agent
ENV_MOCK_MODEL = "NIKA_MOCK_MODEL"

# LLM judge (nika eval judge / nika benchmark run --judge)
ENV_JUDGE_PROVIDER = "NIKA_JUDGE_PROVIDER"
ENV_JUDGE_MODEL = "NIKA_JUDGE_MODEL"


def _env_str(key: str) -> str | None:
    value = os.environ.get(key, "").strip()
    return value or None


def _require_str(*, value: str | None, env_key: str, cli_flag: str) -> str:
    if value:
        return value
    if env := _env_str(env_key):
        return env
    raise ValueError(f"Missing {env_key}: set it in .env or pass {cli_flag}.")


def _require_int(*, value: int | None, env_key: str, cli_flag: str) -> int:
    if value is not None:
        return value
    if raw := _env_str(env_key):
        return int(raw)
    raise ValueError(f"Missing {env_key}: set it in .env or pass {cli_flag}.")


def resolve_agent_type(value: str | None = None) -> str:
    return _require_str(value=value, env_key=ENV_AGENT_TYPE, cli_flag="-a/--agent")


def resolve_llm_provider(value: str | None = None, *, agent_type: str) -> str | None:
    if agent_type.lower() != "react":
        return value
    return _require_str(value=value, env_key=ENV_LLM_PROVIDER, cli_flag="-b/--backend")


def resolve_max_steps(value: int | None = None) -> int:
    if value is not None:
        return value
    if raw := _env_str(ENV_MAX_STEPS):
        return int(raw)
    return 100


def resolve_agent_model(agent_type: str, model: str | None = None) -> str:
    """Resolve model id for *agent_type*; explicit *model* wins over env."""
    if model:
        return model
    if generic := _env_str(ENV_MODEL):
        return generic

    match agent_type.lower():
        case "mock":
            return _require_str(value=None, env_key=ENV_MOCK_MODEL, cli_flag="-m/--model")
        case _:
            return _require_str(value=None, env_key=ENV_REACT_MODEL, cli_flag="-m/--model")


def resolve_judge_provider(value: str | None = None) -> str:
    return _require_str(value=value, env_key=ENV_JUDGE_PROVIDER, cli_flag="-b/--backend (judge)")


def resolve_judge_model(value: str | None = None) -> str:
    return _require_str(value=value, env_key=ENV_JUDGE_MODEL, cli_flag="-m/--model (judge)")
