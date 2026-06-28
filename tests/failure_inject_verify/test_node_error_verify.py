"""Integration tests for CLI failure injection across network-node-error fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/failure_inject_verify/test_node_error_verify.py -v
"""

import unittest

from tests.integration_base import PerTestEnvTestCase


class Bmv2SwitchDownVerifyTest(PerTestEnvTestCase):
    SCENARIO = "p4_counter"

    def test_bmv2_switch_down_verify_true_after_inject(self):
        """failure ps reports injected after bmv2_switch_down."""
        self._inject_via_cli("bmv2_switch_down")
        self._assert_failure_injected("bmv2_switch_down")

    def test_p4_header_error_verify_true_after_inject(self):
        """failure ps reports injected after p4_header_definition_error."""
        self._inject_via_cli("p4_header_definition_error")
        self._assert_failure_injected("p4_header_definition_error")

    def test_p4_parser_state_error_verify_true_after_inject(self):
        """failure ps reports injected after p4_compilation_error_parser_state."""
        self._inject_via_cli("p4_compilation_error_parser_state")
        self._assert_failure_injected("p4_compilation_error_parser_state")

    def test_p4_table_missing_verify_true_after_inject(self):
        """failure ps reports injected after p4_table_entry_missing."""
        self._inject_via_cli("p4_table_entry_missing")
        self._assert_failure_injected("p4_table_entry_missing")

    def test_p4_table_misconfig_verify_true_after_inject(self):
        """failure ps reports injected after p4_table_entry_misconfig."""
        self._inject_via_cli("p4_table_entry_misconfig")
        self._assert_failure_injected("p4_table_entry_misconfig")


class P4MPLSVerifyTest(PerTestEnvTestCase):
    SCENARIO = "p4_mpls"

    def test_p4_mpls_label_limit_exceeded_verify_true_after_inject(self):
        """failure ps reports injected after mpls_label_limit_exceeded."""
        self._inject_via_cli("mpls_label_limit_exceeded")
        self._assert_failure_injected("mpls_label_limit_exceeded")


class FrrDownVerifyTest(PerTestEnvTestCase):
    SCENARIO = "simple_bgp"

    def test_frr_down_verify_true_after_inject(self):
        """failure ps reports injected after frr_service_down."""
        self._inject_via_cli("frr_service_down")
        self._assert_failure_injected("frr_service_down")


class SDNControllerVerifyTest(PerTestEnvTestCase):
    SCENARIO = "sdn_star"
    ENV_RUN_ARGS = ["-s", "s"]

    def test_sdn_controller_crash_verify_true_after_inject(self):
        """failure ps reports injected after sdn_controller_crash."""
        self._inject_via_cli("sdn_controller_crash")
        self._assert_failure_injected("sdn_controller_crash")

    def test_southbound_port_block_verify_true_after_inject(self):
        """failure ps reports injected after southbound_port_block."""
        self._inject_via_cli("southbound_port_block")
        self._assert_failure_injected("southbound_port_block")

    def test_southbound_port_mismatch_verify_true_after_inject(self):
        """failure ps reports injected after southbound_port_mismatch."""
        self._inject_via_cli("southbound_port_mismatch")
        self._assert_failure_injected("southbound_port_mismatch")

    def test_flow_rule_shadowing_verify_true_after_inject(self):
        """failure ps reports injected after flow_rule_shadowing."""
        self._inject_via_cli("flow_rule_shadowing")
        self._assert_failure_injected("flow_rule_shadowing")

    def test_flow_rule_loop_verify_true_after_inject(self):
        """failure ps reports injected after flow_rule_loop."""
        self._inject_via_cli("flow_rule_loop")
        self._assert_failure_injected("flow_rule_loop")


if __name__ == "__main__":
    unittest.main()
