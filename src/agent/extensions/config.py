"""Configuration owned by local extensions, not by NIKA core."""

from __future__ import annotations

import os

from nika.config import RUNTIME_DIR

DEFAULT_LLM_PROVIDER = "custom"
DEFAULT_MODEL = "openai/gpt-oss-120b"

PROCEDURAL_MEMORY_DIR = RUNTIME_DIR / "procedural_memory"
TOOL_REFINEMENT_DIR = RUNTIME_DIR / "tool_refinement"


def default_llm_provider() -> str:
    return (
        os.getenv("NIKA_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
        or DEFAULT_LLM_PROVIDER
    )


def default_model() -> str:
    return os.getenv("NIKA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
