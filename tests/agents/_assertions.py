"""Shared assertions for agent pipeline integration tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.utils.phases import DIAGNOSIS, SUBMISSION


def assert_phase_messages(
    testcase: unittest.TestCase,
    messages: list[dict],
    *,
    require_diagnosis_tools: bool = True,
) -> None:
    agents = {e["agent"] for e in messages}
    testcase.assertIn(DIAGNOSIS, agents)
    testcase.assertIn(SUBMISSION, agents)

    if require_diagnosis_tools:
        diag_tools = [
            e["tool"]["name"]
            for e in messages
            if e["agent"] == DIAGNOSIS and e["event"] == "tool_start" and "tool" in e
        ]
        testcase.assertTrue(diag_tools, "diagnosis phase must call at least one MCP tool")

    sub_tools = [
        e["tool"]["name"]
        for e in messages
        if e["agent"] == SUBMISSION and e["event"] == "tool_start" and "tool" in e
    ]
    testcase.assertIn("list_avail_problems", sub_tools)
    testcase.assertIn("submit", sub_tools)


def assert_submission_fields(testcase: unittest.TestCase, session_dir: Path) -> None:
    submission = json.loads((session_dir / "submission.json").read_text())
    for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
        testcase.assertIn(field, submission)
