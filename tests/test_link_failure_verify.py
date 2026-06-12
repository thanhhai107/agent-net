"""Integration tests for verify_fault across all link-failure fault types.

Each test starts a fresh Kathara lab, injects a real fault, and calls
verify_fault to confirm the environment state matches what was injected.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_link_failure_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.link_failure.link_failure import (
    LinkDetachDetection,
    LinkDetachParams,
    LinkFailureDetection,
    LinkFailureParams,
    LinkFlapDetection,
    LinkFlapParams,
    LinkFragDetection,
    LinkFragParams,
)
from nika.utils.session_store import SessionStore

SCENARIO = "simple_bgp"
HOST = "pc1"
INTF = "eth0"


class LinkFailureVerifyIntegrationTest(unittest.TestCase):
    """Verify that verify_fault correctly reflects real container network state."""

    runner: CliRunner
    session_id: str
    lab_name: str

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

    def _problem(self, cls_):
        """Instantiate a problem class bound to the running lab."""
        return cls_(scenario_name=SCENARIO, lab_name=self.lab_name)

    # ------------------------------------------------------------------
    # LinkFailure: ip link set <intf> down
    # ------------------------------------------------------------------

    def test_link_failure_verify_true_after_inject(self):
        """verify_fault returns verified=True right after ip link set down."""
        params = LinkFailureParams(host_name=HOST, intf_name=INTF)
        problem = self._problem(LinkFailureDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(
            result["verified"],
            f"Expected interface to be down but verify_fault said: {result}",
        )
        self.assertEqual(result["details"]["operstate"], "down")
        self.assertEqual(result["details"]["host"], HOST)
        self.assertEqual(result["details"]["intf"], INTF)

    # ------------------------------------------------------------------
    # LinkFlap: background shell script toggling the interface
    # ------------------------------------------------------------------

    def test_link_flap_verify_true_after_inject(self):
        """verify_fault returns verified=True while the flap script is running."""
        params = LinkFlapParams(host_name=HOST, intf_name=INTF, down_time=30, up_time=30)
        problem = self._problem(LinkFlapDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(
            result["verified"],
            f"Expected flap script to be running but verify_fault said: {result}",
        )
        self.assertEqual(result["details"]["flap_process"], "running")
        self.assertEqual(result["details"]["host"], HOST)
        self.assertEqual(result["details"]["intf"], INTF)

    # ------------------------------------------------------------------
    # LinkDetach: ip link del <intf>
    # ------------------------------------------------------------------

    def test_link_detach_verify_true_after_inject(self):
        """verify_fault returns verified=True after the interface is deleted."""
        params = LinkDetachParams(host_name=HOST, intf_name=INTF)
        problem = self._problem(LinkDetachDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(
            result["verified"],
            f"Expected interface to be absent but verify_fault said: {result}",
        )
        self.assertEqual(result["details"]["host"], HOST)
        self.assertEqual(result["details"]["intf"], INTF)

    # ------------------------------------------------------------------
    # LinkFrag: iptables DROP rule for oversized packets
    # ------------------------------------------------------------------

    def test_link_frag_verify_true_after_inject(self):
        """verify_fault returns verified=True after the frag DROP rule is installed."""
        params = LinkFragParams(host_name=HOST, mtu=100)
        problem = self._problem(LinkFragDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(
            result["verified"],
            f"Expected frag iptables rule to be present but verify_fault said: {result}",
        )


if __name__ == "__main__":
    unittest.main()
