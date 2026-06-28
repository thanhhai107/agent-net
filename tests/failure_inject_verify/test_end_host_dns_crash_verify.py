"""Integration tests for CLI failure injection across dns/crash/vpn/service fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_end_host_dns_crash_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase


class DNSRecordErrorVerifyTest(PerTestEnvTestCase):
    SCENARIO = "ospf_enterprise_dhcp"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_dns_record_error_verify_true_after_inject(self):
        """failure ps reports injected after dns_record_error."""
        self._inject_via_cli("dns_record_error")
        self._assert_failure_injected("dns_record_error")


class HostCrashVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_host_crash_verify_true_after_inject(self):
        """failure ps reports injected after host_crash."""
        self._inject_via_cli("host_crash", {"host_name": "pc1"})
        self._assert_failure_injected("host_crash")


class VPNMembershipMissingVerifyTest(PerTestEnvTestCase):
    SCENARIO = "rip_small_internet_vpn"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_vpn_membership_missing_verify_true_after_inject(self):
        """failure ps reports injected after host_vpn_membership_missing."""
        self._inject_via_cli("host_vpn_membership_missing")
        self._assert_failure_injected("host_vpn_membership_missing")


class ServiceDownVerifyTest(PerTestEnvTestCase):
    """Tests for DNS/DHCP service down via pkill -9 (works in Kathara without systemd)."""

    SCENARIO = "ospf_enterprise_dhcp"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_dns_service_down_verify_true_after_inject(self):
        """failure ps reports injected after dns_service_down."""
        self._inject_via_cli("dns_service_down")
        self._assert_failure_injected("dns_service_down")

    def test_dhcp_service_down_verify_true_after_inject(self):
        """failure ps reports injected after dhcp_service_down."""
        self._inject_via_cli("dhcp_service_down")
        self._assert_failure_injected("dhcp_service_down")


if __name__ == "__main__":
    unittest.main()
