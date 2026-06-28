"""Integration tests for CLI failure injection across link-failure fault types.

Each test starts a fresh Kathara lab, injects a fault via the CLI, and
asserts failure ps reports status=injected.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_link_failure_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase

HOST = "pc1"
INTF = "eth0"
LINK_PARAMS = {"host_name": HOST, "intf_name": INTF}


class LinkFailureVerifyIntegrationTest(PerTestEnvTestCase):
    """Verify CLI inject + failure ps for link failure problems."""

    SCENARIO = "simple_bgp"

    def test_link_failure_verify_true_after_inject(self):
        """failure ps reports injected after link_down."""
        self._inject_via_cli("link_down", LINK_PARAMS)
        self._assert_failure_injected("link_down")

    def test_link_flap_verify_true_after_inject(self):
        """failure ps reports injected after link_flap."""
        self._inject_via_cli("link_flap", LINK_PARAMS)
        self._assert_failure_injected("link_flap")

    def test_link_detach_verify_true_after_inject(self):
        """failure ps reports injected after link_detach."""
        self._inject_via_cli("link_detach", LINK_PARAMS)
        self._assert_failure_injected("link_detach")

    def test_link_frag_verify_true_after_inject(self):
        """failure ps reports injected after link_fragmentation_disabled."""
        self._inject_via_cli("link_fragmentation_disabled", {"host_name": HOST})
        self._assert_failure_injected("link_fragmentation_disabled")


if __name__ == "__main__":
    unittest.main()
