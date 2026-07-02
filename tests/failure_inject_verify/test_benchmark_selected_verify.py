"""Integration tests: inject every case in benchmark_selected.yaml.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_benchmark_selected_verify.py -v
"""

from __future__ import annotations

import os
import unittest

from nika.config import BENCHMARK_DIR
from nika.service.mcp_server.mcp_session_context import SESSION_ID_ENV
from nika.workflows.benchmark.load_config import load_benchmark_yaml

from tests.integration_base import CliIntegrationTestCase


class BenchmarkSelectedInjectVerifyTest(CliIntegrationTestCase):
    """Start a fresh lab per selected benchmark row and verify failure inject."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cases = load_benchmark_yaml(BENCHMARK_DIR / "benchmark_selected.yaml")

    def test_all_selected_cases_inject(self) -> None:
        self.assertEqual(len(self.cases), 56)
        for case in self.cases:
            scenario = case["scenario"]
            problem = case["problem"]
            topo_size = case.get("topo_size") or ""
            inject = case["inject"]
            with self.subTest(scenario=scenario, problem=problem, topo_size=topo_size):
                env_args = ["-s", topo_size] if topo_size else []
                session_id = self._start_env(scenario, env_args)
                prev = os.environ.get(SESSION_ID_ENV)
                os.environ[SESSION_ID_ENV] = session_id
                try:
                    self._assert_session_ready(session_id, scenario)
                    args = ["failure", "inject", problem, "--session_id", session_id]
                    for key, value in inject.items():
                        args += ["--set", f"{key}={value}"]
                    self._invoke_ok(args)
                    self._assert_failure_injected(session_id, problem)
                finally:
                    self._close_session(session_id)
                    if prev is None:
                        os.environ.pop(SESSION_ID_ENV, None)
                    else:
                        os.environ[SESSION_ID_ENV] = prev

    def _assert_failure_injected(self, session_id: str, problem: str) -> None:
        ps_output = self._invoke_ok(["failure", "ps", "--session_id", session_id])
        self.assertIn(f"problem={problem}", ps_output)
        self.assertIn("status=injected", ps_output)


if __name__ == "__main__":
    unittest.main()
