"""Integration tests for CLI failure injection across host-misconfig fault types.

Each test starts a fresh Kathara lab, injects a fault via the CLI, and
asserts failure ps reports status=injected.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_host_misconfig_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase

HOST = "pc1"
HOST2 = "pc2"


class HostMisconfigVerifyIntegrationTest(PerTestEnvTestCase):
    """Verify CLI inject + failure ps for host misconfig problems."""

    SCENARIO = "simple_bgp"

    def test_host_missing_ip_verify_true_after_inject(self):
        """failure ps reports injected after host_missing_ip."""
        self._inject_via_cli("host_missing_ip", {"host_name": HOST, "intf_name": "eth0"})
        self._assert_failure_injected("host_missing_ip")

    def test_host_ip_conflict_verify_true_after_inject(self):
        """failure ps reports injected after host_ip_conflict."""
        self._inject_via_cli("host_ip_conflict", {"host_name": HOST, "host_name_2": HOST2})
        self._assert_failure_injected("host_ip_conflict")

    def test_host_incorrect_ip_verify_true_after_inject(self):
        """failure ps reports injected after host_incorrect_ip."""
        self._inject_via_cli("host_incorrect_ip", {"host_name": HOST})
        self._assert_failure_injected("host_incorrect_ip")

    def test_host_incorrect_gateway_verify_true_after_inject(self):
        """failure ps reports injected after host_incorrect_gateway."""
        self._inject_via_cli("host_incorrect_gateway", {"host_name": HOST})
        self._assert_failure_injected("host_incorrect_gateway")

    def test_host_incorrect_netmask_verify_true_after_inject(self):
        """failure ps reports injected after host_incorrect_netmask."""
        self._inject_via_cli("host_incorrect_netmask", {"host_name": HOST})
        self._assert_failure_injected("host_incorrect_netmask")

    def test_host_incorrect_dns_verify_true_after_inject(self):
        """failure ps reports injected after host_incorrect_dns."""
        self._inject_via_cli("host_incorrect_dns", {"host_name": HOST})
        self._assert_failure_injected("host_incorrect_dns")


if __name__ == "__main__":
    unittest.main()
