"""Session lifecycle and inspection (``nika session``)."""

from nika.workflows.session.close import (
    clean_emulation_environment,
    close_session,
)
from nika.workflows.session.containers import list_session_containers
from nika.workflows.session.inspect import inspect_session
from nika.workflows.session.list import list_sessions

__all__ = [
    "clean_emulation_environment",
    "close_session",
    "inspect_session",
    "list_session_containers",
    "list_sessions",
]
