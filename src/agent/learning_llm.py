"""Shared helpers for bounded learning-module LLM calls."""

from __future__ import annotations

import os

from agent.module_config import module_defaults

ENV_LEARNING_LLM_BACKEND = "NIKA_LEARNING_LLM_BACKEND"
ENV_LEARNING_LLM_MODEL = "NIKA_LEARNING_LLM_MODEL"


def _defaults(module: str):
    defaults = module_defaults()
    if module == "tool_refinement":
        return defaults.tool_refinement
    if module == "procedural_memory":
        return defaults.procedural_memory
    raise ValueError(f"unknown learning module: {module}")


def learning_timeout_seconds(module: str = "procedural_memory") -> float:
    value = os.getenv("NIKA_LEARNING_LLM_TIMEOUT_SECONDS")
    if value is None:
        return _defaults(module).timeout_seconds
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("learning LLM timeout must be a number") from exc
    if parsed <= 0:
        raise ValueError("learning LLM timeout must be positive")
    return parsed


def learning_max_retries(module: str = "procedural_memory") -> int:
    value = os.getenv("NIKA_LEARNING_LLM_MAX_RETRIES")
    if value is None:
        return _defaults(module).max_retries
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("learning LLM retries must be an integer") from exc
    if parsed < 0:
        raise ValueError("learning LLM retries must not be negative")
    return parsed


def learning_backend(
    default: str | None, module: str = "procedural_memory"
) -> str | None:
    value = os.getenv(ENV_LEARNING_LLM_BACKEND)
    if value is not None:
        return value.strip() or default
    return _defaults(module).llm_backend or default


def learning_model(
    default: str | None, module: str = "procedural_memory"
) -> str | None:
    value = os.getenv(ENV_LEARNING_LLM_MODEL)
    if value is not None:
        return value.strip() or default
    return _defaults(module).llm_model or default


def format_learning_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
