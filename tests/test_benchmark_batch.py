"""Parallel benchmark integration test via --batch-size.

Verifies that ``nika benchmark run --batch-size N`` runs N YAML rows simultaneously,
each using the mock agent for Diagnosis, followed by per-session eval—all without
cross-contamination between sessions.

Pipeline per row (all N rows run concurrently within a batch):
  benchmark run (env → inject → mock-agent diagnosis → close → eval metrics)

Assertions:
  - Session IDs are unique across all parallel sessions
  - Each session writes results to its own isolated directory
  - ground_truth.json: is_anomaly=True, root_cause_name contains the injected problem
  - run.json: session_id, scenario_name, agent_type=="mock", status=="finished"
  - submission.json: required fields present, written under the correct session dir
  - eval_metrics.json: required fields present; detection_score==1.0, rca_accuracy==1.0
  - messages.jsonl: both diagnosis and submission phases appear with expected tool calls
  - Runtime session files are removed after benchmark close (scenario undeployed)

Coverage note:
  Running N items with --batch-size N executes them all in parallel via one subprocess
  each, then waits for the batch to complete before returning.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_benchmark_batch.py -v
"""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple

import yaml

from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.config import BENCHMARK_DIR
from nika.utils.session_store import SESSIONS_DIR, SessionStore
from nika.workflows.benchmark.load_config import load_benchmark_yaml
from tests.integration_base import CliIntegrationTestCase

_REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _inject_for_case(scenario: str, problem: str, topo_size: str | None) -> dict[str, str]:
    normalized_topo = topo_size or ""
    for yaml_name in ("benchmark_full.yaml", "benchmark_selected.yaml"):
        path = BENCHMARK_DIR / yaml_name
        if not path.is_file():
            continue
        for row in load_benchmark_yaml(path):
            if (
                row["scenario"] == scenario
                and row["problem"] == problem
                and (row.get("topo_size") or "") == normalized_topo
            ):
                return dict(row["inject"])
    raise ValueError(
        f"No benchmark inject entry for scenario={scenario!r}, problem={problem!r}, "
        f"topo_size={topo_size!r}"
    )


class ParallelBenchmarkIntegrationTest(CliIntegrationTestCase):
    """Run all benchmark YAML rows as one parallel batch, then verify per-session results."""

    _pipeline_results: dict[str, tuple[str, Path] | BaseException]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
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
                    "inject": _inject_for_case(case.scenario, case.problem, case.size),
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result(self, case: ScenarioCase) -> tuple[str, Path]:
        """Return (session_id, session_dir) or fail the sub-test if the pipeline errored."""
        result = self._pipeline_results.get(_case_key(case))
        if isinstance(result, BaseException):
            self.fail(f"Pipeline for {_case_key(case)} raised: {result}")
        self.assertIsNotNone(result, f"No result recorded for {_case_key(case)}")
        return result  # type: ignore[return-value]

    def _load_json(self, session_dir: Path, filename: str) -> dict:
        path = session_dir / filename
        self.assertTrue(path.exists(), f"{filename} missing in {session_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Session isolation
    # ------------------------------------------------------------------

    def test_session_ids_are_unique(self):
        """Each parallel session must have a distinct ID."""
        ids = [
            self._pipeline_results[_case_key(c)][0]
            for c in SCENARIO_CASES
            if not isinstance(self._pipeline_results.get(_case_key(c)), BaseException)
        ]
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate session IDs: {ids}")

    def test_session_dirs_are_isolated(self):
        """Each session must write results to its own distinct directory."""
        dirs = [
            str(self._pipeline_results[_case_key(c)][1])
            for c in SCENARIO_CASES
            if not isinstance(self._pipeline_results.get(_case_key(c)), BaseException)
        ]
        self.assertEqual(len(dirs), len(set(dirs)), f"Overlapping session dirs: {dirs}")

    # ------------------------------------------------------------------
    # ground_truth.json
    # ------------------------------------------------------------------

    def test_ground_truth_correctness(self):
        """is_anomaly is True and root_cause_name contains the injected problem."""
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                _, session_dir = self._result(case)
                gt = self._load_json(session_dir, "ground_truth.json")
                self.assertTrue(gt["is_anomaly"])
                self.assertIn(case.problem, gt["root_cause_name"])

    # ------------------------------------------------------------------
    # run.json
    # ------------------------------------------------------------------

    def test_run_json_correctness(self):
        """run.json records the correct session_id, scenario, agent, and finished status."""
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                run = self._load_json(session_dir, "run.json")
                self.assertEqual(run["session_id"], session_id)
                self.assertEqual(run["scenario_name"], case.scenario)
                self.assertEqual(run["agent_type"], "mock")
                self.assertEqual(run["status"], "finished")

    def test_session_dir_path_contains_session_id(self):
        """The result directory path must embed the session ID (no path cross-wiring)."""
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                self.assertIn(session_id, str(session_dir))

    # ------------------------------------------------------------------
    # submission.json
    # ------------------------------------------------------------------

    def test_submission_fields_and_isolation(self):
        """submission.json has required fields and is written under the correct session dir."""
        for case in SCENARIO_CASES:
            with self.subTest(scenario=case.scenario, problem=case.problem):
                session_id, session_dir = self._result(case)
                sub = self._load_json(session_dir, "submission.json")
                for field in ("is_anomaly", "faulty_devices", "root_cause_name"):
                    self.assertIn(field, sub, f"Missing field '{field}' in submission.json")
                self.assertIn(
                    session_id,
                    str(session_dir),
                    "submission.json not in expected session dir",
                )

    # ------------------------------------------------------------------
    # eval_metrics.json
    # ------------------------------------------------------------------

    def test_eval_metrics_fields_and_scores(self):
        """eval_metrics.json has required fields; detection and RCA accuracy equal 1.0."""
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
                    self.assertIn(field, metrics, f"Missing field '{field}' in eval_metrics.json")
                self.assertEqual(metrics["detection_score"], 1.0)
                self.assertEqual(metrics["rca_accuracy"], 1.0)
                self.assertGreater(metrics["tool_calls"], 0)
                self.assertFalse((session_dir / "llm_judge.json").exists())

    # ------------------------------------------------------------------
    # messages.jsonl (agent trace)
    # ------------------------------------------------------------------

    def test_messages_trace_has_expected_tool_calls(self):
        """messages.jsonl must record MCP tool calls for both diagnosis and submission agents."""
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

    # ------------------------------------------------------------------
    # Runtime session cleanup / scenario undeploy
    # ------------------------------------------------------------------

    def test_runtime_session_files_cleared_after_close(self):
        """After benchmark close the runtime session JSON file must be deleted.

        The benchmark pipeline calls close_session() inside eval_results(), which
        tears down the Kathara lab (undeploys the scenario) and removes the runtime
        file.  Asserting the file is gone confirms each scenario was properly
        undeployed without interfering with the others.
        """
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


def _case_key(case: ScenarioCase) -> str:
    return f"{case.scenario}:{case.problem}"


if __name__ == "__main__":
    unittest.main()
