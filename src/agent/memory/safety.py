"""Safety checks for persistent procedural skills."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


ORACLE_KEYS = {"ground_truth", "answer", "oracle", "expected_root_cause"}
_REDACTED = "[redacted]"

@lru_cache(maxsize=1)
def _known_root_cause_ids() -> tuple[str, ...]:
    try:
        from nika.problems.prob_pool import list_avail_problem_names

        return tuple(
            sorted(
                {str(item).strip() for item in list_avail_problem_names() if str(item).strip()},
                key=len,
                reverse=True,
            )
        )
    except Exception:
        return ()


def redact_oracle_markers(value: Any) -> str:
    """Redact oracle markers and known answer ids before text reaches prompts."""
    text = str(value or "")
    redacted = text
    for marker in ORACLE_KEYS:
        redacted = re.sub(re.escape(marker), _REDACTED, redacted, flags=re.IGNORECASE)
    for root_cause_id in _known_root_cause_ids():
        redacted = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(root_cause_id)}(?![A-Za-z0-9_])",
            _REDACTED,
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted
