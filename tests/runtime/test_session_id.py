"""Unit tests for session id generation."""

from __future__ import annotations

import unittest

from nika.utils.session_id import (
    TEST_SESSION_TAG,
    make_session_id,
    resolve_session_tag,
    session_id_pattern,
)


class MakeSessionIdTest(unittest.TestCase):
    def test_without_tag(self) -> None:
        session_id = make_session_id(suffix="abc123")
        self.assertTrue(session_id.endswith("-abc123"))
        self.assertEqual(session_id.count("-"), 2)

    def test_with_test_tag(self) -> None:
        session_id = make_session_id(session_tag="test", suffix="abc123")
        self.assertRegex(session_id, r"^\d{8}-\d{6}-test-abc123$")

    def test_rejects_invalid_tag(self) -> None:
        with self.assertRaises(ValueError):
            make_session_id(session_tag="BadTag", suffix="abc123")


class ResolveSessionTagTest(unittest.TestCase):
    def test_default_context_no_tag(self) -> None:
        self.assertIsNone(resolve_session_tag())
        self.assertIsNone(resolve_session_tag(context="default"))

    def test_test_context_defaults_to_test(self) -> None:
        self.assertEqual(resolve_session_tag(context="test"), TEST_SESSION_TAG)

    def test_explicit_overrides_test_context(self) -> None:
        self.assertEqual(
            resolve_session_tag("bench", context="test"),
            "bench",
        )

    def test_explicit_overrides_default_context(self) -> None:
        self.assertEqual(resolve_session_tag("bench"), "bench")


class SessionIdPatternTest(unittest.TestCase):
    def test_without_tag(self) -> None:
        pattern = session_id_pattern()
        self.assertRegex("20260707-172930-abc123", pattern)
        self.assertNotRegex("20260707-172930-test-abc123", pattern)

    def test_with_tag(self) -> None:
        pattern = session_id_pattern("test")
        self.assertRegex("20260707-172930-test-abc123", pattern)
        self.assertNotRegex("20260707-172930-abc123", pattern)


if __name__ == "__main__":
    unittest.main()
