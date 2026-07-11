"""Configuration owned by local extensions, not by NIKA core."""

from __future__ import annotations

import os

from nika.config import RUNTIME_DIR

DEFAULT_LLM_PROVIDER = "custom"
DEFAULT_MODEL = "openai/gpt-oss-20b"

MEMORY_DIR = RUNTIME_DIR / "memory"
TOOL_EVOLUTION_DIR = RUNTIME_DIR / "tool_evolution"


def default_llm_provider() -> str:
    return os.getenv("NIKA_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip() or DEFAULT_LLM_PROVIDER


def default_model() -> str:
    return os.getenv("NIKA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

