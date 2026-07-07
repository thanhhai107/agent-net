"""Parallel benchmark integration test via --batch-size.

Verifies that ``nika benchmark run --batch-size N`` runs N YAML rows simultaneously,
each using the mock agent for Diagnosis, followed by per-session eval—all without
cross-contamination between sessions.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests.benchmark.test_batch -v
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple

import yaml

from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session_store import SESSIONS_DIR, SessionStore
from tests.benchmark.helpers import inject_params_from_benchmark_yaml
from tests.integration_base import IntegrationTestCase

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_DONE_RE = re.compile(
    r"benchmark_done session_id=(\S+) scenario=(\S+) problem=(\S+) session_dir=(\S+)"
)


class ScenarioCase(NamedTuple):
    scenario: str
    problem: str
    size: str | None = None


SCENARIO_CASES: list[ScenarioCase] = [
    ScenarioCase("simple_bgp", "link_down"),
    ScenarioCase("simple_bgp", "link_flap"),
    ScenarioCase("simple_bgp", "link_detach"),
    ScenarioCase("ospf_enterprise_dhcp", "dhcp_service_down", size="s"),
    ScenarioCase("rip_small_internet_vpn", "host_vpn_membership_missing", size="s"),
    ScenarioCase("dc_clos_bgp", "bgp_asn_misconfig", size="s"),
    ScenarioCase("ospf_enterprise_dhcp", "dns_record_error", size="s"),
    ScenarioCase("dc_clos_bgp", "host_crash", size="s"),
    ScenarioCase("dc_clos_bgp", "link_fragmentation_disabled", size="s"),
    ScenarioCase("dc_clos_bgp", "bgp_blackhole_route_leak", size="s"),
]


def _case_key(case: ScenarioCase) -> str:
    return f"{case.scenario}:{case.problem}"


class ParallelBenchmarkIntegrationTest(IntegrationTestCase):
    """Run all benchmark YAML rows as one parallel batch, then verify per-session results."""

    _pipeline_results: dict[str, tuple[str, Path] | BaseException]

    @classmethod
    def setUpClass(cls) -> None:
        cls._pipeline_results = {}

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as handle:
            cases = []
            for case in SCENARIO_CASES:
                row = {
                    "scenario": case.scenario,
                    "problem": case.problem,
                    "topo_size": case.size,
                    "inject": inject_params_from_benchmark_yaml(
                        case.scenario,
                        case.problem,
                        case.size or "",
                    ),
                }
                cases.append(row)
            yaml.dump({"cases": cases}, handle, sort_keys=False, allow_unicode=True)
            yaml_path = handle.name

        try:
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "nika",
                    "benchmark",
                    "run",
                    "--config",
                    yaml_path,
                    "--batch-size",
                    str(len(SCENARIO_CASES)),
                    "--agent",
                    "mock",
                    "--model",
                    "mock-v1",
                    "-n",
                    "5",
                ],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
            )
            output = proc.stdout
            if proc.stderr:
                output += proc.stderr
            if proc.returncode != 0:
                raise RuntimeError(
                    f"`nika benchmark run --batch-size {len(SCENARIO_CASES)}` "
                    f"exited {proc.returncode}:\n{output}"
                )

            parsed: dict[str, tuple[str, Path]] = {}
            for match in _BENCHMARK_DONE_RE.finditer(output):
                session_id, scenario, problem, session_dir = match.groups()
                parsed[f"{scenario}:{problem}"] = (session_id, Path(session_dir))

            for case in SCENARIO_CASES:
                key = _case_key(case)
                if key not in parsed:
                    cls._pipeline_results[key] = RuntimeError(
                        f"benchmark_done line missing for {key} in output:\n{output}"
                    )
                else:
                    cls._pipeline_results[key] = parsed[key]
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        for result in cls._pipeline_results.values():
            if isinstance(result, tuple):
                cls._remove_session_results(result[0])

    def _result(self, case: ScenarioCase) -> tuple[str, Path]:
        result = self._pipeline_results.get(_case_key(case))
        if isinstance(result, BaseException):
            self.fail(f"Pipeline for {_case_key(case)} raised: {result}")
        self.assertIsNotNone(result, f"No result recorded for {_case_key(case)}")
        return result  # type: ignore[return-value]

    def _load_json(self, session_dir: Path, filename: str) -> dict:
        path = session_dir / filename
        self.assertTrue(path.exists(), f"{filename} missing in {session_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_session_ids_are_unique(self) -> None:
        ids = [
            self._pipeline_results[_case_key(c)][0]
            for c in SCENARIO_CASES
            if not isinstance(self._pipeline_results.get(_case_key(c)), BaseException)
        ]
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate session IDs: {ids}")

    def test_session_dirs_are_isolated(self) -> None:
        dirs = [
            str(self._pipeline_results[_case_key(c)][1])
            for c in SCENARIO_CASES
            if not isinstance(self._pipeline_results.get(_case_key(c)), BaseException)
        ]
        self.assertEqual(len(dirs), len(set(dirs)), f"Overlapping session dirs: {dirs}")

    def test_ground_truth_correctness(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                _, session_dir = self._result(case)
                gt = self._load_json(session_dir, "ground_truth.json")
                self.assertTrue(gt["is_anomaly"])
                self.assertIn(case.problem, gt["root_cause_name"])

    def test_run_json_correctness(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                run = self._load_json(session_dir, "run.json")
                self.assertEqual(run["session_id"], session_id)
                self.assertEqual(run["scenario_name"], case.scenario)
                self.assertEqual(run["agent_type"], "mock")
                self.assertEqual(run["status"], "finished")

    def test_session_dir_path_contains_session_id(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                self.assertIn(session_id, str(session_dir))

    def test_submission_fields_and_isolation(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                sub = self._load_json(session_dir, "submission.json")
                for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
                    self.assertIn(
                        field, sub, f"Missing field '{field}' in submission.json"
                    )
                self.assertIn(session_id, str(session_dir))

    def test_eval_metrics_fields_and_scores(self) -> None:
        required_fields = (
            "detection_score",
            "localization_accuracy",
            "localization_f1",
            "rca_accuracy",
            "rca_f1",
            "tool_calls",
        )
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                _, session_dir = self._result(case)
                metrics = self._load_json(session_dir, "eval_metrics.json")
                for field in required_fields:
                    self.assertIn(
                        field, metrics, f"Missing field '{field}' in eval_metrics.json"
                    )
                self.assertEqual(metrics["detection_score"], 1.0)
                self.assertEqual(metrics["rca_accuracy"], 1.0)
                self.assertGreater(metrics["tool_calls"], 0)
                self.assertFalse((session_dir / "llm_judge.json").exists())

    def test_messages_trace_has_expected_tool_calls(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                _, session_dir = self._result(case)
                trace_path = session_dir / "messages.jsonl"
                self.assertTrue(trace_path.exists(), "messages.jsonl missing")

                events = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                agents_seen = {e["agent"] for e in events}
                self.assertIn(DIAGNOSIS, agents_seen)
                self.assertIn(SUBMISSION, agents_seen)

                tool_names_seen = {
                    e["tool"]["name"]
                    for e in events
                    if e.get("event") == "tool_start" and "tool" in e
                }
                self.assertIn("list_avail_problems", tool_names_seen)
                self.assertIn("submit", tool_names_seen)

    def test_runtime_session_files_cleared_after_close(self) -> None:
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, _ = self._result(case)
                runtime_path = Path(SESSIONS_DIR) / f"{session_id}.json"
                self.assertFalse(
                    runtime_path.exists(),
                    f"Runtime session file was not removed after undeploy: {runtime_path}",
                )
                with self.assertRaises(FileNotFoundError):
                    SessionStore().get_session(session_id)


if __name__ == "__main__":
    unittest.main()
