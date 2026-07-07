"""Shared helpers for live backend API smoke tests."""

from __future__ import annotations

import asyncio
import json
import shutil
import unittest
from collections.abc import Callable
from typing import Any

import docker


def docker_available() -> bool:
    try:
        docker.from_env().ping()
    except Exception:
        return False
    return True


def min3clos_prerequisites() -> bool:
    if not docker_available():
        return False
    return bool(shutil.which("clab") and shutil.which("gnmic"))


def docker_image_available(image: str) -> bool:
    if not docker_available():
        return False
    try:
        return bool(docker.from_env().images.list(name=image))
    except Exception:
        return False


def p4_int_prerequisites() -> bool:
    """``p4_int`` needs the custom Influx collector image."""
    return docker_image_available("kathara/influxdb")


class ApiSmokeMixin:
    """Mixin that records API smoke calls and fails on parse/runtime errors."""

    def smoke(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        expect_type: type | tuple[type, ...] | None = None,
        min_len: int = 0,
    ) -> Any:
        try:
            result = fn()
        except json.JSONDecodeError as exc:
            self.fail(f"{label}: JSON parse error: {exc}")
        except (ValueError, RuntimeError, TypeError) as exc:
            self.fail(f"{label}: {type(exc).__name__}: {exc}")

        if expect_type is not None:
            self.assertIsInstance(
                result,
                expect_type,
                f"{label}: expected {expect_type}, got {type(result)}",
            )
        if min_len > 0:
            text = "" if result is None else str(result)
            self.assertGreaterEqual(
                len(text),
                min_len,
                f"{label}: unexpected empty result ({result!r})",
            )
        return result

    def smoke_async(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        min_len: int = 0,
    ) -> Any:
        return self.smoke(label, lambda: asyncio.run(fn()), min_len=min_len)


def assert_json_payload(test: unittest.TestCase, label: str, payload: str) -> dict:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        test.fail(f"{label}: invalid JSON: {exc}\n{payload!r}")
    test.assertIsInstance(parsed, dict, f"{label}: JSON root must be an object")
    return parsed
