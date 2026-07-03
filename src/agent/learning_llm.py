"""Shared helpers for bounded learning-module LLM calls."""

from __future__ import annotations

import os

DEFAULT_LEARNING_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LEARNING_LLM_MAX_RETRIES = 0
ENV_LEARNING_LLM_BACKEND = "NIKA_LEARNING_LLM_BACKEND"
ENV_LEARNING_LLM_MODEL = "NIKA_LEARNING_LLM_MODEL"


def learning_timeout_seconds() -> float:
    value = os.getenv("NIKA_LEARNING_LLM_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_LEARNING_LLM_TIMEOUT_SECONDS
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("NIKA_LEARNING_LLM_TIMEOUT_SECONDS must be a number") from exc
    if parsed <= 0:
        raise ValueError("NIKA_LEARNING_LLM_TIMEOUT_SECONDS must be greater than 0")
    return parsed


def learning_max_retries() -> int:
    value = os.getenv("NIKA_LEARNING_LLM_MAX_RETRIES")
    if value is None:
        return DEFAULT_LEARNING_LLM_MAX_RETRIES
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("NIKA_LEARNING_LLM_MAX_RETRIES must be an integer") from exc
    if parsed < 0:
        raise ValueError("NIKA_LEARNING_LLM_MAX_RETRIES must be greater than or equal to 0")
    return parsed


def learning_backend(default: str | None) -> str | None:
    value = os.getenv(ENV_LEARNING_LLM_BACKEND)
    if value is None:
        return default
    return value.strip() or None


def learning_model(default: str | None) -> str | None:
    value = os.getenv(ENV_LEARNING_LLM_MODEL)
    if value is None:
        return default
    return value.strip() or None


def format_learning_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
