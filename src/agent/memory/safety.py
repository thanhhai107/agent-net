"""Safety helpers for benchmark-safe procedural memory."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

FORBIDDEN_MEMORY_KEYS = {
    "problem",
    "problem_name",
    "problem_names",
    "ground_truth",
    "ground_truth_path",
    "faulty_devices_from_ground_truth",
    "failure_injections",
    "injection_params",
    "verify_result",
    "requested_overrides",
    "resolved_params",
}

_FORBIDDEN_JSON_KEY = re.compile(
    r'["\'](' + "|".join(re.escape(key) for key in FORBIDDEN_MEMORY_KEYS) + r')["\']\s*:',
    re.IGNORECASE,
)
_GROUND_TRUTH_FILE = re.compile(r"\bground_truth\.json\b", re.IGNORECASE)


class MemoryOracleLeakageError(ValueError):
    """Raised when a memory payload contains benchmark-oracle fields."""


def find_oracle_leaks(payload: Any, *, path: str = "$") -> list[str]:
    """Return paths that appear to expose oracle-only benchmark data."""

    leaks: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.lower() in FORBIDDEN_MEMORY_KEYS:
                leaks.append(child_path)
            leaks.extend(find_oracle_leaks(value, path=child_path))
    elif isinstance(payload, str):
        if _FORBIDDEN_JSON_KEY.search(payload) or _GROUND_TRUTH_FILE.search(payload):
            leaks.append(path)
    elif isinstance(payload, Sequence) and not isinstance(payload, bytes | bytearray):
        for index, value in enumerate(payload):
            leaks.extend(find_oracle_leaks(value, path=f"{path}[{index}]"))
    return leaks


def assert_no_oracle_leakage(payload: Any) -> None:
    """Fail closed before sending a payload to a memory LLM or embedding index."""

    leaks = find_oracle_leaks(payload)
    if leaks:
        joined = ", ".join(leaks[:8])
        if len(leaks) > 8:
            joined += f", ... (+{len(leaks) - 8} more)"
        raise MemoryOracleLeakageError(
            f"memory payload contains oracle-only fields: {joined}"
        )
