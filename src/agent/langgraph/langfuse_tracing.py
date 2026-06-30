"""Best-effort Langfuse callback setup for LangGraph workflows."""

from __future__ import annotations

from collections.abc import Sequence

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from nika.utils.logger import system_logger


def create_langfuse_callbacks() -> list[CallbackHandler]:
    """Return a Langfuse callback when auth succeeds, otherwise keep running."""
    try:
        handler = CallbackHandler()
        if get_client().auth_check():
            system_logger.info("Authentication to Langfuse successful.")
            return [handler]
        system_logger.warning(
            "Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY."
        )
    except Exception as exc:
        system_logger.warning(
            "Langfuse tracing disabled because auth check failed: %s",
            exc,
        )
    return []


def callback_config(callbacks: Sequence[object]) -> dict[str, list[object]]:
    """Build a LangChain config fragment only when callbacks are available."""
    return {"callbacks": list(callbacks)} if callbacks else {}
