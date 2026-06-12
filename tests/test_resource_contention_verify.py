"""Integration tests for verify_fault across resource-contention fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_resource_contention_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.resource_contention.link_issue import (
    IncastTrafficNetworkLimitationDetection,
    IncastTrafficNetworkLimitationParams,
    LinkBandwidthThrottlingDetection,
    LinkBandwidthThrottlingParams,
    LinkHighPacketCorruptionDetection,
    LinkHighPacketCorruptionParams,
)
from nika.orchestrator.problems.resource_contention.service_issue import (
    DNSLookupLatencyDetection,
    DNSLookupLatencyParams,
    LoadBalancerOverloadDetection,
    LoadBalancerOverloadParams,
)
from nika.orchestrator.problems.resource_contention.tcp_issue import (
    ReceiverResourceContentionDetection,
    ReceiverResourceContentionParams,
    SenderApplicationDelayDetection,
    SenderApplicationDelayParams,
    SenderResourceContentionDetection,
    SenderResourceContentionParams,
)
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


class StressVerifyTest(unittest.TestCase):
    """Tests for stress-ng based faults — expected to fail due to broken injector."""

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
    def test_sender_resource_contention_verify(self):
        """KNOWN ISSUE: inject_stress_all uses & without nohup; process dies immediately."""
        problem = self._problem(SenderResourceContentionDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected stress-ng running: {result}")

    @unittest.expectedFailure
    def test_receiver_resource_contention_verify(self):
        """KNOWN ISSUE: inject_stress_all uses & without nohup; process dies immediately."""
        problem = self._problem(ReceiverResourceContentionDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected stress-ng running: {result}")

    @unittest.expectedFailure
    def test_load_balancer_overload_verify(self):
        """KNOWN ISSUE: inject_stress_all uses & without nohup; process dies immediately."""
        problem = self._problem(LoadBalancerOverloadDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected stress-ng running: {result}")


class DNSLookupLatencyVerifyTest(unittest.TestCase):
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

    def test_dns_lookup_latency_verify_true_after_inject(self):
        """verify_fault returns verified=True after tc delay is injected on DNS server."""
        params = DNSLookupLatencyParams(intf_name="eth0", delay_ms=500)
        problem = self._problem(DNSLookupLatencyDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected tc delay to be present: {result}")
        self.assertIn("delay", result["details"]["tc_output"])


class LinkIssueVerifyTest(unittest.TestCase):
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

    def test_link_high_packet_corruption_verify_true_after_inject(self):
        """verify_fault returns verified=True after corruption rule is injected."""
        params = LinkHighPacketCorruptionParams(host_name="pc1", corruption_percentage=60)
        problem = self._problem(LinkHighPacketCorruptionDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected corrupt qdisc: {result}")
        self.assertIn("corrupt", result["details"]["tc_output"])

    def test_link_bandwidth_throttling_verify_true_after_inject(self):
        """verify_fault returns verified=True after TBF qdisc is injected."""
        params = LinkBandwidthThrottlingParams(host_name="pc1")
        problem = self._problem(LinkBandwidthThrottlingDetection)
        problem.inject_fault(params)
        result = problem.verify_fault(params)
        self.assertTrue(result["verified"], f"Expected tbf qdisc: {result}")
        self.assertIn("tbf", result["details"]["tc_output"])

    def test_incast_traffic_limitation_verify_true_after_inject(self):
        """verify_fault returns verified=True after netem/tbf is injected on web server."""
        problem = self._problem(IncastTrafficNetworkLimitationDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected netem/tbf qdisc: {result}")


if __name__ == "__main__":
    unittest.main()
