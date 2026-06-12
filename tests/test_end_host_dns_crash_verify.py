"""Integration tests for verify_fault across dns/crash/vpn/service fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_end_host_dns_crash_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.end_host_failure.dns import DNSRecordErrorDetection, DNSRecordErrorParams
from nika.orchestrator.problems.end_host_failure.host_failure import HostCrashDetection, HostCrashParams
from nika.orchestrator.problems.end_host_failure.service_failure import (
    DHCPServiceDownDetection,
    DHCPServiceDownParams,
    DNSServiceDownDetection,
    DNSServiceDownParams,
)
from nika.orchestrator.problems.end_host_failure.vpn import HostIncorrectDNSDetection as VPNMembershipMissingDetection
from nika.orchestrator.problems.end_host_failure.vpn import VPNMembershipMissingParams
from nika.utils.session_store import SessionStore


class DNSRecordErrorVerifyTest(unittest.TestCase):
    SCENARIO = "ospf_enterprise_dhcp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", self.SCENARIO])
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
            app_runner = CliRunner()
            app_runner.invoke(app, ["env", "stop", "--session-id", self.session_id])

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_dns_record_error_verify_true_after_inject(self):
        """verify_fault returns verified=True after DNS zone file is modified."""
        problem = self._problem(DNSRecordErrorDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected DNS record error to be verified: {result}")
        self.assertIn("found", result["details"]["grep_result"])


class HostCrashVerifyTest(unittest.TestCase):
    SCENARIO = "simple_bgp"
    HOST = "pc1"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", self.SCENARIO])
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
            CliRunner().invoke(app, ["env", "stop", "--session-id", self.session_id])

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_host_crash_verify_true_after_inject(self):
        """verify_fault returns verified=True after host container is paused."""
        params = HostCrashParams(host_name=self.HOST)
        problem = self._problem(HostCrashDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected host crash to be verified: {result}")
        self.assertEqual(result["details"]["container_status"], "paused")


class VPNMembershipMissingVerifyTest(unittest.TestCase):
    SCENARIO = "rip_small_internet_vpn"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", self.SCENARIO])
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
            CliRunner().invoke(app, ["env", "stop", "--session-id", self.session_id])

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_vpn_membership_missing_verify_true_after_inject(self):
        """verify_fault returns verified=True after VPN membership is commented out."""
        problem = self._problem(VPNMembershipMissingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected VPN membership missing to be verified: {result}")


class ServiceDownVerifyTest(unittest.TestCase):
    """Tests for DNS/DHCP service down — expected to fail due to Kathara systemctl no-op."""

    SCENARIO = "ospf_enterprise_dhcp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", self.SCENARIO])
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
            CliRunner().invoke(app, ["env", "stop", "--session-id", self.session_id])

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    @unittest.expectedFailure
    def test_dns_service_down_verify_true_after_inject(self):
        """KNOWN ISSUE: systemctl stop named is no-op in Kathara; process won't stop."""
        problem = self._problem(DNSServiceDownDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected DNS service to be down: {result}")

    @unittest.expectedFailure
    def test_dhcp_service_down_verify_true_after_inject(self):
        """KNOWN ISSUE: systemctl stop isc-dhcp-server is no-op in Kathara; process won't stop."""
        problem = self._problem(DHCPServiceDownDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected DHCP service to be down: {result}")


if __name__ == "__main__":
    unittest.main()
