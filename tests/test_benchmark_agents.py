"""Benchmark integration tests for supported agent types.

Only ``mock`` and LangGraph ``react`` are covered here.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

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


@dataclass(frozen=True)
class AgentBenchmarkConfig:
    agent_type: str
    extra_args: tuple[str, ...]
    diagnosis_phase: str
    submission_phase: str
    expect_perfect_scores: bool


AGENT_CONFIGS: dict[str, AgentBenchmarkConfig] = {
    "mock": AgentBenchmarkConfig(
        agent_type="mock",
        extra_args=("-a", "mock", "-b", "openai", "-m", "mock-v1", "-n", "5"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=False,
    ),
    "react": AgentBenchmarkConfig(
        agent_type="react",
        extra_args=("-a", "react", "-b", "deepseek", "-m", "deepseek-chat", "-n", "20"),
        diagnosis_phase=DIAGNOSIS,
        submission_phase=SUBMISSION,
        expect_perfect_scores=False,
    ),
}


def _run_benchmark(config: AgentBenchmarkConfig) -> tuple[str, Path, str]:
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
        raise RuntimeError(f"`{' '.join(cmd)}` exited {proc.returncode}:\n{output}")

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
            self.assertRegex(self.pipeline_output, _BENCHMARK_DONE_RE.pattern)

        def test_ground_truth(self) -> None:
            gt = self._load_json("ground_truth.json")
            self.assertTrue(gt["is_anomaly"])
            self.assertIn(PROBLEM, gt["root_cause_name"])

        def test_run_json(self) -> None:
            run = self._load_json("run.json")
            self.assertEqual(run["session_id"], self.session_id)
            self.assertEqual(run["scenario_name"], SCENARIO)
            self.assertEqual(run["agent_type"], config.agent_type)
            self.assertEqual(run["status"], "finished")

        def test_submission_json(self) -> None:
            sub = self._load_json("submission.json")
            for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
                self.assertIn(field, sub, f"Missing field '{field}' in submission.json")

        def test_eval_metrics(self) -> None:
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
            events = self._load_messages()
            agents_seen = {e["agent"] for e in events}
            self.assertIn(config.diagnosis_phase, agents_seen)
            self.assertIn(config.submission_phase, agents_seen)
            tool_names = {
                e["tool"]["name"]
                for e in events
                if e.get("event") == "tool_start" and "tool" in e
            }
            self.assertIn("list_avail_problems", tool_names)
            self.assertIn("submit", tool_names)

        def test_runtime_session_cleared(self) -> None:
            runtime_path = Path(SESSIONS_DIR) / f"{self.session_id}.json"
            self.assertFalse(runtime_path.exists(), f"Runtime session file remains: {runtime_path}")
            with self.assertRaises(FileNotFoundError):
                SessionStore().get_session(self.session_id)

    AgentBenchmarkTest.__name__ = f"{config_key.title()}AgentBenchmarkTest"
    AgentBenchmarkTest.__qualname__ = AgentBenchmarkTest.__name__
    AgentBenchmarkTest.__doc__ = f"Benchmark pipeline with the {config.agent_type!r} agent."
    return AgentBenchmarkTest


MockAgentBenchmarkTest = _make_benchmark_agent_test_class("mock")
ReactAgentBenchmarkTest = unittest.skipUnless(
    _deepseek_api_key_available(),
    "DEEPSEEK_API_KEY required for react benchmark (deepseek-chat)",
)(_make_benchmark_agent_test_class("react"))


if __name__ == "__main__":
    unittest.main()
