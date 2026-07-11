"""Shared helpers for network environment verification tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar
from unittest.mock import patch

T = TypeVar("T")


def ready_node_count(output: str) -> int:
    ready = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "Ready":
            ready += 1
    return ready


def instantiate_with_mocked_kathara(
    patch_target: str,
    factory: Callable[[], T],
) -> T:
    with patch(patch_target, return_value=object()):
        return factory()


def assert_verify_success(testcase, result: dict) -> None:
    testcase.assertTrue(result["verified"], result["checks"])
    testcase.assertTrue(all(result["checks"].values()), result["checks"])
