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


def session_problem_label(session: Any) -> str:
    """Return a stable problem label for tracing metadata."""
    problem_names = getattr(session, "problem_names", None) or []
    if problem_names:
        return str(problem_names[0])

    root_cause = getattr(session, "root_cause_name", "") or ""
    if isinstance(root_cause, (list, tuple)):
        return str(root_cause[0]) if root_cause else ""
    return str(root_cause)
