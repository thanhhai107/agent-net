"""Session id formatting helpers."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from uuid import uuid4

SessionTagContext = Literal["default", "test"]

TEST_SESSION_TAG = "test"
_SESSION_TAG_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def resolve_session_tag(
    explicit: str | None = None,
    *,
    context: SessionTagContext = "default",
) -> str | None:
    """Pick a session id tag.

    Explicit values (e.g. benchmark ``--session-tag``) override context defaults.
    Normal runs use no tag; test contexts default to ``test``.
    """
    if explicit is not None:
        return explicit
    if context == "test":
        return TEST_SESSION_TAG
    return None


def session_id_pattern(tag: str | None = None) -> re.Pattern[str]:
    """Return a regex matching session ids, optionally with a fixed tag."""
    if tag is None:
        return re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
    return re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(tag)}-[0-9a-f]{{6}}$")


def make_session_id(
    *, session_tag: str | None = None, suffix: str | None = None
) -> str:
    """Build a session id: ``YYYYMMDD-HHMMSS[-tag]-{6hex}``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    token = suffix or uuid4().hex[:6]
    if not session_tag:
        return f"{stamp}-{token}"
    if not _SESSION_TAG_RE.fullmatch(session_tag):
        raise ValueError(
            "session_tag must start with a letter and contain only "
            "lowercase letters, digits, underscores, or hyphens."
        )
    return f"{stamp}-{session_tag}-{token}"
