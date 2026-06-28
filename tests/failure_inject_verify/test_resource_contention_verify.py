"""Integration tests for CLI failure injection across resource-contention fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_resource_contention_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase


class StressVerifyTest(PerTestEnvTestCase):
    """Tests for stress-ng based faults."""

    SCENARIO = "ospf_enterprise_dhcp"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_sender_resource_contention_verify(self):
        """failure ps reports injected after sender_resource_contention."""
        self._inject_via_cli("sender_resource_contention")
        self._assert_failure_injected("sender_resource_contention")

    def test_receiver_resource_contention_verify(self):
        """failure ps reports injected after receiver_resource_contention."""
        self._inject_via_cli("receiver_resource_contention")
        self._assert_failure_injected("receiver_resource_contention")

    def test_load_balancer_overload_verify(self):
        """failure ps reports injected after load_balancer_overload."""
        self._inject_via_cli("load_balancer_overload")
        self._assert_failure_injected("load_balancer_overload")


class DNSLookupLatencyVerifyTest(PerTestEnvTestCase):
    SCENARIO = "dc_clos_service"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_dns_lookup_latency_verify_true_after_inject(self):
        """failure ps reports injected after dns_lookup_latency."""
        self._inject_via_cli("dns_lookup_latency")
        self._assert_failure_injected("dns_lookup_latency")


class LinkIssueVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_link_high_packet_corruption_verify_true_after_inject(self):
        """failure ps reports injected after link_high_packet_corruption."""
        self._inject_via_cli("link_high_packet_corruption", {"host_name": "pc1"})
        self._assert_failure_injected("link_high_packet_corruption")

    def test_link_bandwidth_throttling_verify_true_after_inject(self):
        """failure ps reports injected after link_bandwidth_throttling."""
        self._inject_via_cli("link_bandwidth_throttling", {"host_name": "pc1"})
        self._assert_failure_injected("link_bandwidth_throttling")


class IncastTrafficLimitationVerifyTest(PerTestEnvTestCase):
    """Incast limitation targets web servers; requires dc_clos_service, not simple_bgp."""

    SCENARIO = "dc_clos_service"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_incast_traffic_limitation_verify_true_after_inject(self):
        """failure ps reports injected after incast_traffic_network_limitation."""
        self._inject_via_cli("incast_traffic_network_limitation")
        self._assert_failure_injected("incast_traffic_network_limitation")


if __name__ == "__main__":
    unittest.main()
