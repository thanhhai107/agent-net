"""Tests for unified agent trace metrics."""

from __future__ import annotations

import json
from pathlib import Path

from nika.evaluator.trace_parser import AgentTraceParser


def test_default_trace_parser_counts_all_diagnosis_agent_tags(tmp_path: Path) -> None:
    trace = tmp_path / "messages.jsonl"
    rows = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "agent": "diagnosis_agent",
            "event": "tool_start",
            "tool": {"name": "ping_pair"},
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "agent": "diagnosis_agent",
            "event": "llm_end",
            "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "timestamp": "2026-01-01T00:00:02",
            "agent": "memory_agent",
            "event": "skill_transition",
            "tool": "ping_pair",
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    metrics = AgentTraceParser(str(trace)).parse_trace()

    assert metrics["tool_calls"] == 1
    assert metrics["steps"] == 1
    assert metrics["in_tokens"] == 10
    assert metrics["out_tokens"] == 5

