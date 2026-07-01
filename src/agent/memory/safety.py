"""Safety checks for persistent procedural skills."""

from __future__ import annotations

from typing import Any


class MemoryOracleLeakageError(ValueError):
    pass


ORACLE_KEYS = {"ground_truth", "answer", "oracle", "expected_root_cause"}


def assert_no_oracle_leakage(payload: Any) -> None:
    text = str(payload).lower()
    for key in ORACLE_KEYS:
        if key in text:
            raise MemoryOracleLeakageError(f"Potential oracle leakage marker: {key}")
