"""Integration tests for verify_fault across misconfiguration fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_misconfig_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.misconfigurations.acl_error import (
    ARPAclBlockDetection,
    ARPAclBlockParams,
    BGPAclBlockDetection,
    BGPAclBlockParams,
    DNSPortBlockedDetection,
    DNSPortBlockedParams,
    HttpAclBlockDetection,
    HttpAclBlockParams,
    IcmpAclBlockDetection,
    IcmpAclBlockParams,
    OSPFAclBlockDetection,
    OSPFAclBlockParams,
)
from nika.orchestrator.problems.misconfigurations.bgp import (
    BGPAsnMisconfigDetection,
    BGPAsnMisconfigParams,
    BGPBlackholeRouteLeakDetection,
    BGPHijackingDetection,
    BGPHijackingParams,
    BGPMissingAdvertiseDetection,
    BGPMissingAdvertiseParams,
    StaticBlackHoleDetection,
    StaticBlackHoleParams,
)
from nika.orchestrator.problems.misconfigurations.dhcp import DHCPMissingSubnetDetection, DHCPMissingSubnetParams
from nika.orchestrator.problems.misconfigurations.mac import MacAddressConflictDetection, MacAddressConflictParams
from nika.orchestrator.problems.misconfigurations.ospf import (
    OSPFAreaMisconfigDetection,
    OSPFAreaMisconfigParams,
    OSPFNeighborMissingDetection,
    OSPFNeighborMissingParams,
)
from nika.orchestrator.problems.misconfigurations.p4 import (
    P4AggressiveDetectionThresholdsDetection,
    P4AggressiveDetectionThresholdsParams,
)
from nika.utils.session_store import SessionStore


def _setup_env(runner, scenario, **kwargs):
    args = ["env", "run", scenario]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    result = runner.invoke(app, args)
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


class OSPFMisconfigVerifyTest(unittest.TestCase):
    SCENARIO = "ospf_enterprise_static"

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
    def test_ospf_area_misconfig_verify(self):
        """KNOWN ISSUE: systemctl restart is no-op in Kathara; in-memory config unchanged."""
        problem = self._problem(OSPFAreaMisconfigDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected OSPF area misconfig: {result}")

    def test_ospf_neighbor_missing_verify_file_updated(self):
        """verify_fault checks frr.conf file has commented network lines (not in-memory)."""
        problem = self._problem(OSPFNeighborMissingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected OSPF neighbor missing (file): {result}")
        self.assertGreater(result["details"]["commented_network_count"], 0)


class BGPMisconfigVerifyTest(unittest.TestCase):
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

    @unittest.expectedFailure
    def test_bgp_asn_misconfig_verify(self):
        """KNOWN ISSUE: systemctl restart is no-op in Kathara; in-memory ASN unchanged."""
        problem = self._problem(BGPAsnMisconfigDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected BGP ASN misconfig: {result}")

    @unittest.expectedFailure
    def test_bgp_missing_advertise_verify(self):
        """KNOWN ISSUE: sed \\1 escape bug + systemctl no-op."""
        problem = self._problem(BGPMissingAdvertiseDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected BGP missing advertise: {result}")

    def test_static_blackhole_verify_true_after_inject(self):
        """verify_fault returns verified=True after blackhole route is injected."""
        problem = self._problem(StaticBlackHoleDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected blackhole route: {result}")
        self.assertIn("blackhole", result["details"]["route_output"])

    def test_bgp_blackhole_route_leak_verify(self):
        """verify_fault returns verified=True after Null0 route is added."""
        problem = self._problem(BGPBlackholeRouteLeakDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected Null0 route: {result}")

    def test_bgp_hijacking_verify_true_after_inject(self):
        """verify_fault returns verified=True after BGP hijacking advertisement is added."""
        problem = self._problem(BGPHijackingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected BGP hijacking: {result}")


class MacMisconfigVerifyTest(unittest.TestCase):
    SCENARIO = "ospf_enterprise_static"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_mac_address_conflict_verify_true_after_inject(self):
        """verify_fault returns verified=True after MAC address conflict is injected."""
        problem = self._problem(MacAddressConflictDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected MAC conflict: {result}")
        self.assertEqual(result["details"]["mac_0"].lower(), result["details"]["mac_1"].lower())


class DHCPMisconfigVerifyTest(unittest.TestCase):
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

    def test_dhcp_missing_subnet_verify_true_after_inject(self):
        """verify_fault returns verified=True after DHCP subnet is deleted."""
        problem = self._problem(DHCPMissingSubnetDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected DHCP missing subnet: {result}")
        self.assertIn("absent", result["details"]["grep_result"])


class ACLBlockVerifyTest(unittest.TestCase):
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

    def test_bgp_acl_block_verify_true_after_inject(self):
        """verify_fault returns verified=True after BGP ACL block is injected."""
        problem = self._problem(BGPAclBlockDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected BGP ACL block: {result}")

    def test_icmp_acl_block_verify_true_after_inject(self):
        """verify_fault returns verified=True after ICMP ACL block is injected."""
        problem = self._problem(IcmpAclBlockDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected ICMP ACL block: {result}")

    def test_http_acl_block_verify_true_after_inject(self):
        """verify_fault returns verified=True after HTTP ACL block is injected."""
        problem = self._problem(HttpAclBlockDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected HTTP ACL block: {result}")

    def test_arp_acl_block_verify_true_after_inject(self):
        """verify_fault returns verified=True after ARP ACL block is injected."""
        problem = self._problem(ARPAclBlockDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected ARP ACL block: {result}")


class P4MisconfigVerifyTest(unittest.TestCase):
    SCENARIO = "p4_bloom_filter"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_p4_aggressive_detection_thresholds_verify_true_after_inject(self):
        """verify_fault returns verified=True after PACKET_THRESHOLD is reduced."""
        problem = self._problem(P4AggressiveDetectionThresholdsDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected P4 threshold modified: {result}")
        self.assertTrue(result["details"]["threshold_modified"])


if __name__ == "__main__":
    unittest.main()
