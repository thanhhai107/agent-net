"""Integration tests for verify_fault across all host-misconfig fault types.

Each test starts a fresh Kathara lab, injects a real fault, and calls
verify_fault to confirm the environment state matches what was injected.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_host_misconfig_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.end_host_failure.host_misconfig import (
    HostIPConflictDetection,
    HostIPConflictParams,
    HostIncorrectDNSDetection,
    HostIncorrectDNSParams,
    HostIncorrectGatewayDetection,
    HostIncorrectGatewayParams,
    HostIncorrectIPDetection,
    HostIncorrectIPParams,
    HostIncorrectNetmaskDetection,
    HostIncorrectNetmaskParams,
    HostMissingIPDetection,
    HostMissingIPParams,
)
from nika.utils.session_store import SessionStore

SCENARIO = "simple_bgp"
HOST = "pc1"
HOST2 = "pc2"


class HostMisconfigVerifyIntegrationTest(unittest.TestCase):
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
        return cls_(scenario_name=SCENARIO, lab_name=self.lab_name)

    # ------------------------------------------------------------------
    # HostMissingIP
    # ------------------------------------------------------------------

    def test_host_missing_ip_verify_true_after_inject(self):
        """verify_fault returns verified=True after IP is removed."""
        params = HostMissingIPParams(host_name=HOST, intf_name="eth0")
        problem = self._problem(HostMissingIPDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected missing IP to be verified: {result}")
        self.assertEqual(result["details"]["host"], HOST)
        self.assertNotIn("inet ", result["details"]["ip_line"])

    # ------------------------------------------------------------------
    # HostIPConflict
    # ------------------------------------------------------------------

    def test_host_ip_conflict_verify_true_after_inject(self):
        """verify_fault returns verified=True after IP conflict is injected."""
        params = HostIPConflictParams(host_name=HOST, host_name_2=HOST2)
        problem = self._problem(HostIPConflictDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected IP conflict to be verified: {result}")
        self.assertEqual(result["details"]["ip_a"], result["details"]["ip_b"])

    # ------------------------------------------------------------------
    # HostIncorrectIP
    # ------------------------------------------------------------------

    def test_host_incorrect_ip_verify_true_after_inject(self):
        """verify_fault returns verified=True after incorrect IP is injected."""
        params = HostIncorrectIPParams(host_name=HOST)
        problem = self._problem(HostIncorrectIPDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected incorrect IP to be verified: {result}")
        self.assertIn("inet 10.2.1.", result["details"]["ip_line"])

    # ------------------------------------------------------------------
    # HostIncorrectGateway
    # ------------------------------------------------------------------

    def test_host_incorrect_gateway_verify_true_after_inject(self):
        """verify_fault returns verified=True after incorrect gateway is injected."""
        params = HostIncorrectGatewayParams(host_name=HOST)
        problem = self._problem(HostIncorrectGatewayDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected incorrect gateway to be verified: {result}")
        self.assertIn(".254", result["details"]["route_line"])

    # ------------------------------------------------------------------
    # HostIncorrectNetmask
    # ------------------------------------------------------------------

    def test_host_incorrect_netmask_verify_true_after_inject(self):
        """verify_fault returns verified=True after incorrect netmask is injected."""
        params = HostIncorrectNetmaskParams(host_name=HOST, netmask_prefix=8)
        problem = self._problem(HostIncorrectNetmaskDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected incorrect netmask to be verified: {result}")
        self.assertNotEqual(result["details"]["actual_prefix"], 24)

    # ------------------------------------------------------------------
    # HostIncorrectDNS
    # ------------------------------------------------------------------

    def test_host_incorrect_dns_verify_true_after_inject(self):
        """verify_fault returns verified=True after incorrect DNS resolver is injected."""
        params = HostIncorrectDNSParams(host_name=HOST, fake_dns_ip="8.8.8.8")
        problem = self._problem(HostIncorrectDNSDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected incorrect DNS to be verified: {result}")
        self.assertIn("8.8.8.8", result["details"]["resolv_conf"])


if __name__ == "__main__":
    unittest.main()
