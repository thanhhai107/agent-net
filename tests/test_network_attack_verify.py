"""Integration tests for verify_fault across network-attack fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_network_attack_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.network_under_attack.arp import ArpCachePoisoningDetection, ArpCachePoisoningParams
from nika.orchestrator.problems.network_under_attack.bgp import BGPHijackingDetection, BGPHijackingParams
from nika.orchestrator.problems.network_under_attack.dhcp import (
    DHCPSpoofedDNSDetection,
    DHCPSpoofedDNSParams,
    DHCPSpoofedGatewayDetection,
    DHCPSpoofedGatewayParams,
    DHCPSpoofedSubnetDetection,
    DHCPSpoofedSubnetParams,
)
from nika.orchestrator.problems.network_under_attack.web import WebDoSDetection, WebDoSParams
from nika.utils.session_store import SessionStore


def _setup_env(runner, scenario):
    result = runner.invoke(app, ["env", "run", scenario])
    if result.exit_code != 0:
        raise RuntimeError(f"nika env run failed:\n{result.output}")
    match = re.search(r"session_id=(\S+)", result.output.strip())
    if match is None:
        raise RuntimeError(f"session_id not found in env run output:\n{result.output}")
    session_id = match.group(1)
    row = SessionStore().get_session(session_id)
    return session_id, row["lab_name"]


def _teardown_env(runner, session_id):
    runner.invoke(app, ["env", "stop", "--session-id", session_id])


class WebDoSVerifyTest(unittest.TestCase):
    SCENARIO = "dc_clos_service"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    @unittest.expectedFailure
    def test_web_dos_verify_true_after_inject(self):
        """KNOWN ISSUE: inject_ab_attack uses & which may not survive exec_cmd session."""
        problem = self._problem(WebDoSDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected ab attack running: {result}")


class DHCPAttackVerifyTest(unittest.TestCase):
    SCENARIO = "ospf_enterprise_dhcp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    @unittest.expectedFailure
    def test_dhcp_spoofed_gateway_verify(self):
        """KNOWN ISSUE: dhclient may overwrite the fault on clients."""
        problem = self._problem(DHCPSpoofedGatewayDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected spoofed gateway: {result}")

    @unittest.expectedFailure
    def test_dhcp_spoofed_dns_verify(self):
        """KNOWN ISSUE: dhclient may overwrite the fault on clients."""
        problem = self._problem(DHCPSpoofedDNSDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected spoofed DNS: {result}")

    def test_dhcp_spoofed_subnet_verify(self):
        """verify_fault returns verified=True after subnet is deleted from dhcpd.conf."""
        problem = self._problem(DHCPSpoofedSubnetDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected subnet deleted: {result}")


class BGPHijackingVerifyTest(unittest.TestCase):
    SCENARIO = "simple_bgp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_bgp_hijacking_verify_true_after_inject(self):
        """verify_fault returns verified=True after BGP hijacking is injected."""
        problem = self._problem(BGPHijackingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected BGP hijacking to be verified: {result}")
        self.assertTrue(result["details"]["has_advertisement"])

    def test_arp_cache_poisoning_verify_true_after_inject(self):
        """verify_fault returns verified=True after ARP cache is poisoned."""
        params = ArpCachePoisoningParams(host_name="pc1", fake_mac="00:11:22:33:44:55")
        problem = ArpCachePoisoningDetection(scenario_name=self.SCENARIO, lab_name=self.lab_name)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected ARP poisoning to be verified: {result}")
        self.assertIn("00:11:22:33:44:55", result["details"]["neigh_entry"])


if __name__ == "__main__":
    unittest.main()
