"""Integration tests for CLI failure injection across misconfiguration fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_misconfig_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase


class OSPFMisconfigVerifyTest(PerTestEnvTestCase):
    SCENARIO = "ospf_enterprise_static"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_ospf_area_misconfig_verify(self):
        """failure ps reports injected after ospf_area_misconfiguration."""
        self._inject_via_cli("ospf_area_misconfiguration")
        self._assert_failure_injected("ospf_area_misconfiguration")

    def test_ospf_neighbor_missing_verify_file_updated(self):
        """failure ps reports injected after ospf_neighbor_missing."""
        self._inject_via_cli("ospf_neighbor_missing")
        self._assert_failure_injected("ospf_neighbor_missing")


class BGPMisconfigVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_bgp_asn_misconfig_verify(self):
        """failure ps reports injected after bgp_asn_misconfig."""
        self._inject_via_cli("bgp_asn_misconfig")
        self._assert_failure_injected("bgp_asn_misconfig")

    def test_bgp_missing_advertise_verify(self):
        """failure ps reports injected after bgp_missing_route_advertisement."""
        self._inject_via_cli("bgp_missing_route_advertisement")
        self._assert_failure_injected("bgp_missing_route_advertisement")

    def test_static_blackhole_verify_true_after_inject(self):
        """failure ps reports injected after host_static_blackhole."""
        self._inject_via_cli("host_static_blackhole")
        self._assert_failure_injected("host_static_blackhole")

    def test_bgp_blackhole_route_leak_verify(self):
        """failure ps reports injected after bgp_blackhole_route_leak."""
        self._inject_via_cli("bgp_blackhole_route_leak")
        self._assert_failure_injected("bgp_blackhole_route_leak")

    def test_bgp_hijacking_verify_true_after_inject(self):
        """failure ps reports injected after bgp_hijacking."""
        self._inject_via_cli("bgp_hijacking")
        self._assert_failure_injected("bgp_hijacking")


class MacMisconfigVerifyTest(PerTestEnvTestCase):
    SCENARIO = "ospf_enterprise_static"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_mac_address_conflict_verify_true_after_inject(self):
        """failure ps reports injected after mac_address_conflict."""
        self._inject_via_cli("mac_address_conflict")
        self._assert_failure_injected("mac_address_conflict")


class DHCPMisconfigVerifyTest(PerTestEnvTestCase):
    SCENARIO = "ospf_enterprise_dhcp"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_dhcp_missing_subnet_verify_true_after_inject(self):
        """failure ps reports injected after dhcp_missing_subnet."""
        self._inject_via_cli("dhcp_missing_subnet")
        self._assert_failure_injected("dhcp_missing_subnet")


class ACLBlockVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_bgp_acl_block_verify_true_after_inject(self):
        """failure ps reports injected after bgp_acl_block."""
        self._inject_via_cli("bgp_acl_block")
        self._assert_failure_injected("bgp_acl_block")

    def test_icmp_acl_block_verify_true_after_inject(self):
        """failure ps reports injected after icmp_acl_block."""
        self._inject_via_cli("icmp_acl_block")
        self._assert_failure_injected("icmp_acl_block")

    def test_http_acl_block_verify_true_after_inject(self):
        """failure ps reports injected after http_acl_block."""
        self._inject_via_cli("http_acl_block")
        self._assert_failure_injected("http_acl_block")

    def test_arp_acl_block_verify_true_after_inject(self):
        """failure ps reports injected after arp_acl_block."""
        self._inject_via_cli("arp_acl_block")
        self._assert_failure_injected("arp_acl_block")


class P4MisconfigVerifyTest(PerTestEnvTestCase):
    SCENARIO = "p4_bloom_filter"

    def test_p4_aggressive_detection_thresholds_verify_true_after_inject(self):
        """failure ps reports injected after p4_aggressive_detection_thresholds."""
        self._inject_via_cli("p4_aggressive_detection_thresholds")
        self._assert_failure_injected("p4_aggressive_detection_thresholds")


if __name__ == "__main__":
    unittest.main()
