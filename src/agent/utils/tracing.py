"""Observability helpers with Langfuse as the default tracing path."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import langsmith as ls


def _env_enabled(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def langsmith_tracing_context(
    *,
    project_name: str = "NIKA",
    metadata: dict[str, Any] | None = None,
):
    """Return a LangSmith context only when explicitly enabled.

    Langfuse callbacks are the default observability path. This keeps LangSmith
    optional and avoids noisy client warnings when LangSmith env vars are unset
    or intentionally left blank.
    """
    if not _env_enabled("LANGSMITH_TRACING", default=False):
        return nullcontext()
    return ls.tracing_context(project_name=project_name, metadata=metadata)
