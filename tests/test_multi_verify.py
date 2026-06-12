"""Integration tests for verify_fault in MultiFaultBase.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_multi_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.end_host_failure.host_misconfig import (
    HostMissingIPDetection,
    HostMissingIPParams,
)
from nika.orchestrator.problems.link_failure.link_failure import LinkFailureDetection, LinkFailureParams
from nika.orchestrator.problems.multi_problems import MultiFaultDetection
from nika.utils.session_store import SessionStore

SCENARIO = "simple_bgp"


class MultiFaultVerifyTest(unittest.TestCase):
    """Verify that MultiFaultBase.verify_fault aggregates sub-fault results."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", SCENARIO])
        if result.exit_code != 0:
            raise RuntimeError(f"nika env run failed:\n{result.output}")
        match = re.search(r"session_id=(\S+)", result.output.strip())
        if match is None:
            raise RuntimeError(f"session_id not found in env run output:\n{result.output}")
        self.session_id = match.group(1)
        row = SessionStore().get_session(self.session_id)
        self.lab_name = row["lab_name"]

    def tearDown(self) -> None:
        if getattr(self, "session_id", None):
            self.runner.invoke(app, ["env", "stop", "--session-id", self.session_id])

    def test_multi_fault_verify_aggregates_sub_results(self):
        """MultiFaultDetection.verify_fault returns aggregated sub-fault results."""
        link_fault = LinkFailureDetection(scenario_name=SCENARIO, lab_name=self.lab_name)
        host_fault = HostMissingIPDetection(scenario_name=SCENARIO, lab_name=self.lab_name)

        multi = MultiFaultDetection(
            sub_faults=[link_fault, host_fault],
            scenario_name=SCENARIO,
            lab_name=self.lab_name,
        )

        link_params = LinkFailureParams(host_name="pc1", intf_name="eth0")
        host_params = HostMissingIPParams(host_name="pc2", intf_name="eth0")
        link_fault.inject_fault(link_params)
        host_fault.inject_fault(host_params)

        result = multi.verify_fault()

        self.assertIn("verified", result)
        self.assertIn("sub_results", result["details"])
        self.assertEqual(len(result["details"]["sub_results"]), 2)

        sub_results = result["details"]["sub_results"]
        self.assertTrue(sub_results[0]["verified"], f"Expected link_fault verified: {sub_results[0]}")
        self.assertTrue(sub_results[1]["verified"], f"Expected host_fault verified: {sub_results[1]}")
        self.assertTrue(result["verified"], f"Expected all sub-faults verified: {result}")


if __name__ == "__main__":
    unittest.main()
