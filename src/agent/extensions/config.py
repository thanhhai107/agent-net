"""Configuration owned by local extensions, not by NIKA core."""

from __future__ import annotations

from agent.module_config import module_defaults
from nika.config import RUNTIME_DIR

DEFAULT_LLM_PROVIDER = module_defaults().baseline.llm_provider
DEFAULT_MODEL = module_defaults().baseline.model

PROCEDURAL_MEMORY_DIR = RUNTIME_DIR / "procedural_memory"
TOOL_REFINEMENT_DIR = RUNTIME_DIR / "tool_refinement"


def default_llm_provider() -> str:
    return DEFAULT_LLM_PROVIDER


def default_model() -> str:
    return DEFAULT_MODEL
