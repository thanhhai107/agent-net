"""Agent CLI configuration from environment variables.

No hard-coded defaults: each setting must come from a CLI flag or ``.env``.
CLI flags take precedence when both are set.
"""

from __future__ import annotations

import os

from agent.module_config import module_defaults

# Shared CLI options (nika agent run / nika benchmark run)
ENV_AGENT_TYPE = "NIKA_AGENT_TYPE"
ENV_LLM_PROVIDER = "NIKA_LLM_PROVIDER"
ENV_MAX_STEPS = "NIKA_MAX_STEPS"
ENV_MODEL = "NIKA_MODEL"

# LLM judge (nika eval judge / nika benchmark run --judge)
ENV_JUDGE_PROVIDER = "NIKA_JUDGE_PROVIDER"
ENV_JUDGE_MODEL = "NIKA_JUDGE_MODEL"

SUPPORTED_AGENT_TYPES = ("react", "plan-execute", "reflexion")


def _env_str(key: str) -> str | None:
    value = os.environ.get(key, "").strip()
    return value or None


def canonical_agent_type(agent_type: str) -> str:
    """Return the public agent identifier used in session artifacts.

    ``byo.langgraph`` was the original implementation-facing identifier.  Keep
    accepting it for existing commands, but persist the product-facing ReAct
    identifier everywhere else.
    """
    normalized = agent_type.strip().lower()
    if normalized == "byo.langgraph":
        return "react"
    if normalized in {"plan_execute", "plan-and-execute"}:
        return "plan-execute"
    return normalized


def resolve_agent_type(value: str | None = None) -> str:
    configured = value or module_defaults().baseline.agent_type
    agent_type = canonical_agent_type(configured)
    if agent_type not in SUPPORTED_AGENT_TYPES:
        supported = ", ".join(SUPPORTED_AGENT_TYPES)
        raise ValueError(
            f"Unsupported agent type {configured!r}; choose one of: {supported}"
        )
    return agent_type


def resolve_llm_provider(value: str | None = None, *, agent_type: str) -> str | None:
    if canonical_agent_type(agent_type) not in SUPPORTED_AGENT_TYPES:
        return value
    return value or module_defaults().baseline.llm_provider


def resolve_max_steps(value: int | None = None) -> int:
    return value if value is not None else module_defaults().baseline.max_steps


def resolve_agent_model(agent_type: str, model: str | None = None) -> str:
    """Resolve the model from an explicit value or shared baseline config."""
    if model:
        return model
    if canonical_agent_type(agent_type) in SUPPORTED_AGENT_TYPES:
        return module_defaults().baseline.model

    normalized = canonical_agent_type(agent_type)
    if normalized in SUPPORTED_AGENT_TYPES:
        return module_defaults().baseline.model
    if normalized == "mock":
        return _env_str(ENV_MODEL) or "mock"
    raise ValueError(f"Unsupported agent type for model resolution: {agent_type!r}")


def resolve_judge_provider(value: str | None = None) -> str:
    return value or module_defaults().baseline.judge_provider


def resolve_judge_model(value: str | None = None) -> str:
    return value or module_defaults().baseline.judge_model
