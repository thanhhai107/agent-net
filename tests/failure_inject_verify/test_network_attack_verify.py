"""Integration tests for CLI failure injection across network-attack fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_network_attack_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase


class WebDoSVerifyTest(PerTestEnvTestCase):
    SCENARIO = "dc_clos_service"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_web_dos_verify_true_after_inject(self):
        """failure ps reports injected after web_dos_attack."""
        self._inject_via_cli("web_dos_attack")
        self._assert_failure_injected("web_dos_attack")


class DHCPAttackVerifyTest(PerTestEnvTestCase):
    SCENARIO = "ospf_enterprise_dhcp"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_dhcp_spoofed_gateway_verify(self):
        """failure ps reports injected after dhcp_spoofed_gateway."""
        self._inject_via_cli("dhcp_spoofed_gateway")
        self._assert_failure_injected("dhcp_spoofed_gateway")

    def test_dhcp_spoofed_dns_verify(self):
        """failure ps reports injected after dhcp_spoofed_dns."""
        self._inject_via_cli("dhcp_spoofed_dns")
        self._assert_failure_injected("dhcp_spoofed_dns")

    def test_dhcp_spoofed_subnet_verify(self):
        """failure ps reports injected after dhcp_spoofed_subnet."""
        self._inject_via_cli("dhcp_spoofed_subnet")
        self._assert_failure_injected("dhcp_spoofed_subnet")


class BGPHijackingVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_bgp_hijacking_verify_true_after_inject(self):
        """failure ps reports injected after bgp_hijacking."""
        self._inject_via_cli("bgp_hijacking")
        self._assert_failure_injected("bgp_hijacking")

    def test_arp_cache_poisoning_verify_true_after_inject(self):
        """failure ps reports injected after arp_cache_poisoning."""
        self._inject_via_cli("arp_cache_poisoning", {"host_name": "pc1"})
        self._assert_failure_injected("arp_cache_poisoning")


if __name__ == "__main__":
    unittest.main()
