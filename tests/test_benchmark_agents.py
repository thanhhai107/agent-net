"""Benchmark integration tests for all implemented agent types.

Verifies that ``mock``, ``react``, ``codex_cli``, and ``claude_cli`` agents can complete
the full benchmark pipeline on a real Kathara scenario with an injected failure:

    nika benchmark run <scenario> --problem <problem> -a <agent> ...

Each agent type is exercised via the same benchmark entry point used in batch
CSV runs (``src/nika/workflows/benchmark/run.py``), not the step-by-step CLI
pipeline used in ``test_pipeline.py`` / ``test_integration_agents.py``.

Assertions (common):
  - ``benchmark_done`` line emitted with session_id and session_dir
  - ground_truth.json: is_anomaly=True, root_cause_name matches injected problem
  - run.json: correct scenario, agent_type, status=finished
  - submission.json: required fields present
  - eval_metrics.json: required fields present; tool_calls > 0
  - messages.jsonl: diagnosis and submission phase agents logged

Agent-specific:
  - mock: detection_score and rca_accuracy == 1.0
  - codex_cli: codex_workspace/ present; codex events in messages.jsonl
  - claude_cli: claude_workspace/ present; claude stream-json events in messages.jsonl

Prerequisites:
  - Docker must be running
  - mock: always runs (no LLM)
  - react: DEEPSEEK_API_KEY in ``.env`` (provider ``deepseek``, model ``deepseek-chat``)
  - codex_cli: Codex CLI installed and authenticated (``codex login`` or OPENAI_API_KEY)
  - claude_cli: Claude Code CLI installed (``claude`` in PATH); credentials via
    ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` in ``.env`` or ``claude auth login``
  - claude model: ``ANTHROPIC_MODEL`` (or related env vars) when ``-m`` is omitted

Run:
    uv run python -m unittest tests/test_benchmark_agents.py -v

Run a single agent class:
    uv run python -m unittest tests.test_benchmark_agents.MockAgentBenchmarkTest -v
    uv run python -m unittest tests.test_benchmark_agents.ReactAgentBenchmarkTest -v
    uv run python -m unittest tests.test_benchmark_agents.CodexCliAgentBenchmarkTest -v
    uv run python -m unittest tests.test_benchmark_agents.ClaudeCliAgentBenchmarkTest -v
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

from agent.claude_cli.config import claude_credentials_available, default_claude_model
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session_store import SESSIONS_DIR, SessionStore
from tests.integration_base import CliIntegrationTestCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")
_BENCHMARK_DONE_RE = re.compile(
    r"benchmark_done session_id=(\S+) scenario=(\S+) problem=(\S+) session_dir=(\S+)"
)

SCENARIO = "simple_bgp"
PROBLEM = "link_down"


def _deepseek_api_key_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _openai_api_key_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _codex_cli_available() -> bool:
    if shutil.which("codex") is None:
        return False
    return _openai_api_key_available() or (Path.home() / ".codex" / "auth.json").is_file()


def _claude_cli_available() -> bool:
    return claude_credentials_available()


@dataclass(frozen=True)
class AgentBenchmarkConfig:
    agent_type: str
    extra_args: tuple[str, ...]
    diagnosis_phase: str
    submission_phase: str
    expect_perfect_scores: bool
    workspace_dir: str | None = None
    subprocess_event_key: str | None = None


AGENT_CONFIGS: dict[str, AgentBenchmarkConfig] = {
    "mock": AgentBenchmarkConfig(
        agent_type="mock",
        extra_args=("-a", "mock", "-p", "mock", "-m", "mock-v1", "-n", "5"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=True,
    ),
    "react": AgentBenchmarkConfig(
        agent_type="react",
        extra_args=("-a", "react", "-p", "deepseek", "-m", "deepseek-chat", "-n", "20"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=False,
    ),
    "codex_cli": AgentBenchmarkConfig(
        agent_type="codex_cli",
        extra_args=("-a", "codex_cli", "-m", "gpt-5.4-mini"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=False,
        workspace_dir="codex_workspace",
        subprocess_event_key="codex_event",
    ),
    "claude_cli": AgentBenchmarkConfig(
        agent_type="claude_cli",
        extra_args=("-a", "claude_cli", "-p", "anthropic"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=False,
        workspace_dir="claude_workspace",
        subprocess_event_key="claude_event",
    ),
}


def _run_benchmark(config: AgentBenchmarkConfig) -> tuple[str, Path, str]:
    """Run one benchmark case and return (session_id, session_dir, combined_output)."""
    cmd = [
        "uv",
        "run",
        "nika",
        "benchmark",
        "run",
        SCENARIO,
        "--problem",
        PROBLEM,
        *config.extra_args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = proc.stdout
    if proc.stderr:
        output += proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` exited {proc.returncode}:\n{output}"
        )

    match = _BENCHMARK_DONE_RE.search(output)
    if match is None:
        raise RuntimeError(f"benchmark_done line missing from output:\n{output}")

    session_id, scenario, problem, session_dir = match.groups()
    if scenario != SCENARIO or problem != PROBLEM:
        raise RuntimeError(
            f"benchmark_done mismatch: expected {SCENARIO}/{PROBLEM}, "
            f"got {scenario}/{problem}\n{output}"
        )
    return session_id, Path(session_dir), output


def _make_benchmark_agent_test_class(config_key: str) -> type[CliIntegrationTestCase]:
    """Build a concrete unittest class for one agent benchmark configuration."""
    config = AGENT_CONFIGS[config_key]

    class AgentBenchmarkTest(CliIntegrationTestCase):
        session_id: ClassVar[str]
        session_dir: ClassVar[Path]
        pipeline_output: ClassVar[str]

        @classmethod
        def setUpClass(cls) -> None:
            super().setUpClass()
            cls.session_id, cls.session_dir, cls.pipeline_output = _run_benchmark(config)

        @classmethod
        def tearDownClass(cls) -> None:
            if getattr(cls, "session_id", None):
                cls._remove_session_results(cls.session_id)

        def _load_json(self, filename: str) -> dict:
            path = self.session_dir / filename
            self.assertTrue(path.exists(), f"{filename} missing in {self.session_dir}")
            return json.loads(path.read_text(encoding="utf-8"))

        def _load_messages(self) -> list[dict]:
            trace_path = self.session_dir / "messages.jsonl"
            self.assertTrue(trace_path.exists(), "messages.jsonl missing")
            return [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        def test_benchmark_done_emitted(self) -> None:
            """Benchmark CLI must print a parseable benchmark_done line."""
            self.assertRegex(self.pipeline_output, _BENCHMARK_DONE_RE.pattern)

        def test_ground_truth(self) -> None:
            """Injected failure is recorded in ground_truth.json."""
            gt = self._load_json("ground_truth.json")
            self.assertTrue(gt["is_anomaly"])
            self.assertIn(PROBLEM, gt["root_cause_name"])

        def test_run_json(self) -> None:
            """run.json records the benchmark scenario, agent, and finished status."""
            run = self._load_json("run.json")
            self.assertEqual(run["session_id"], self.session_id)
            self.assertEqual(run["scenario_name"], SCENARIO)
            self.assertEqual(run["agent_type"], config.agent_type)
            self.assertEqual(run["status"], "finished")
            self.assertIn(self.session_id, str(self.session_dir))

        def test_submission_json(self) -> None:
            """submission.json is written with the required schema fields."""
            sub = self._load_json("submission.json")
            for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
                self.assertIn(field, sub, f"Missing field '{field}' in submission.json")

        def test_eval_metrics(self) -> None:
            """eval_metrics.json is produced after benchmark close."""
            metrics = self._load_json("eval_metrics.json")
            for field in (
                "detection_score",
                "localization_accuracy",
                "localization_f1",
                "rca_accuracy",
                "rca_f1",
                "tool_calls",
            ):
                self.assertIn(field, metrics, f"Missing field '{field}' in eval_metrics.json")
            self.assertGreater(metrics["tool_calls"], 0)
            if config.expect_perfect_scores:
                self.assertEqual(metrics["detection_score"], 1.0)
                self.assertEqual(metrics["rca_accuracy"], 1.0)

        def test_messages_trace(self) -> None:
            """messages.jsonl records both diagnosis and submission phases."""
            events = self._load_messages()
            agents_seen = {e["agent"] for e in events}
            self.assertIn(config.diagnosis_phase, agents_seen)
            self.assertIn(config.submission_phase, agents_seen)

            if config.subprocess_event_key == "codex_event":
                mcp_tools = {
                    (e.get("codex_event") or {}).get("item", {}).get("tool")
                    for e in events
                    if "codex_event" in e
                }
                mcp_tools.discard(None)
                self.assertIn("list_avail_problems", mcp_tools)
                self.assertIn("submit", mcp_tools)
            elif config.subprocess_event_key == "claude_event":
                tool_names: set[str] = set()
                for e in events:
                    if "claude_event" not in e:
                        continue
                    content = ((e.get("claude_event") or {}).get("message") or {}).get("content") or []
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_use":
                            continue
                        name = str(block.get("name", ""))
                        if "list_avail_problems" in name:
                            tool_names.add("list_avail_problems")
                        if "submit" in name:
                            tool_names.add("submit")
                self.assertIn("list_avail_problems", tool_names)
                self.assertIn("submit", tool_names)
            else:
                tool_names = {
                    e["tool"]["name"]
                    for e in events
                    if e.get("event") == "tool_start" and "tool" in e
                }
                self.assertIn("list_avail_problems", tool_names)
                self.assertIn("submit", tool_names)

        def test_runtime_session_cleared(self) -> None:
            """Benchmark close removes the runtime session JSON file."""
            runtime_path = Path(SESSIONS_DIR) / f"{self.session_id}.json"
            self.assertFalse(runtime_path.exists(), f"Runtime session file remains: {runtime_path}")
            with self.assertRaises(FileNotFoundError):
                SessionStore().get_session(self.session_id)

        def test_agent_workspace_when_applicable(self) -> None:
            """Subprocess CLI agents must leave an isolated workspace directory."""
            if config.workspace_dir is None:
                for name in ("codex_workspace", "claude_workspace"):
                    self.assertFalse(
                        (self.session_dir / name).exists(),
                        f"unexpected {name}/ for {config.agent_type}",
                    )
                return

            workspace = self.session_dir / config.workspace_dir
            self.assertTrue(workspace.is_dir(), f"{config.workspace_dir}/ must exist for {config.agent_type}")

            if config.agent_type == "codex_cli":
                self.assertTrue((workspace / ".codex_home").is_dir())
            elif config.agent_type == "claude_cli":
                self.assertTrue((workspace / "diagnosis_mcp_config.json").exists())
                self.assertTrue((workspace / "submission_mcp_config.json").exists())

            if config.subprocess_event_key:
                events = self._load_messages()
                subprocess_events = [e for e in events if config.subprocess_event_key in e]
                self.assertGreater(
                    len(subprocess_events),
                    0,
                    f"{config.agent_type} agent must log {config.subprocess_event_key} events",
                )

    AgentBenchmarkTest.__name__ = f"{config_key.title()}AgentBenchmarkTest"
    AgentBenchmarkTest.__qualname__ = AgentBenchmarkTest.__name__
    AgentBenchmarkTest.__doc__ = f"Benchmark pipeline with the {config.agent_type!r} agent."
    return AgentBenchmarkTest


MockAgentBenchmarkTest = _make_benchmark_agent_test_class("mock")
ReactAgentBenchmarkTest = unittest.skipUnless(
    _deepseek_api_key_available(),
    "DEEPSEEK_API_KEY required for react benchmark (deepseek-chat)",
)(_make_benchmark_agent_test_class("react"))
CodexCliAgentBenchmarkTest = unittest.skipUnless(
    _codex_cli_available(),
    "Codex CLI and OpenAI credentials required",
)(_make_benchmark_agent_test_class("codex_cli"))
ClaudeCliAgentBenchmarkTest = unittest.skipUnless(
    _claude_cli_available(),
    f"Claude Code CLI and credentials required (model from env: {default_claude_model()!r})",
)(_make_benchmark_agent_test_class("claude_cli"))


if __name__ == "__main__":
    unittest.main()
