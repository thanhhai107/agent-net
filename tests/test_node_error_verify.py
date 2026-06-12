"""Integration tests for verify_fault across network-node-error fault types.

Prerequisites:
  - Docker must be running
  - Run via: uv run python -m unittest tests/test_node_error_verify.py -v
"""

import re
import unittest

from typer.testing import CliRunner

from nika.cli.main import app
from nika.orchestrator.problems.network_node_error.controller_issues import (
    FlowRuleLoopDetection,
    FlowRuleLoopParams,
    FlowRuleShadowingDetection,
    FlowRuleShadowingParams,
    SDNControllerCrashDetection,
    SDNControllerCrashParams,
    SouthboundPortBlockDetection,
    SouthboundPortBlockParams,
    SouthboundPortMismatchDetection,
    SouthboundPortMismatchParams,
)
from nika.orchestrator.problems.network_node_error.p4_pipeline_error import (
    P4CompilationErrorHeaderDetection,
    P4CompilationErrorParserStateDetection,
    P4MPLSLabelLimitExceededDetection,
    P4TableEntryMisconfigDetection,
    P4TableEntryMissingDetection,
)
from nika.orchestrator.problems.network_node_error.swicth_router_failure import (
    Bmv2SwitchDownDetection,
    Bmv2SwitchDownParams,
    FrrDownDetection,
    FrrDownParams,
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


class Bmv2SwitchDownVerifyTest(unittest.TestCase):
    SCENARIO = "p4_counter"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_bmv2_switch_down_verify_true_after_inject(self):
        """verify_fault returns verified=True after simple_switch is killed."""
        problem = self._problem(Bmv2SwitchDownDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected bmv2 down to be verified: {result}")

    def test_p4_header_error_verify_true_after_inject(self):
        """verify_fault returns verified=True after P4 header is corrupted."""
        problem = self._problem(P4CompilationErrorHeaderDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected P4 header error to be verified: {result}")

    def test_p4_parser_state_error_verify_true_after_inject(self):
        """verify_fault returns verified=True after P4 parser state is corrupted."""
        problem = self._problem(P4CompilationErrorParserStateDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected P4 parser error to be verified: {result}")

    def test_p4_table_missing_verify_true_after_inject(self):
        """verify_fault returns verified=True after table entries are cleared."""
        problem = self._problem(P4TableEntryMissingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected P4 table missing to be verified: {result}")

    def test_p4_table_misconfig_verify_true_after_inject(self):
        """verify_fault returns verified=True after table entries are modified with 66:66: MACs."""
        problem = self._problem(P4TableEntryMisconfigDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected P4 table misconfig to be verified: {result}")


class P4MPLSVerifyTest(unittest.TestCase):
    SCENARIO = "p4_mpls"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        self.session_id, self.lab_name = _setup_env(self.runner, self.SCENARIO)

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_p4_mpls_label_limit_exceeded_verify_true_after_inject(self):
        """verify_fault returns verified=True after MPLS label limit is reduced."""
        problem = self._problem(P4MPLSLabelLimitExceededDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected MPLS label limit exceeded to be verified: {result}")
        self.assertTrue(result["details"]["const_modified"])


class FrrDownVerifyTest(unittest.TestCase):
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
    def test_frr_down_verify_true_after_inject(self):
        """KNOWN ISSUE: systemctl stop frr is no-op in Kathara; ospfd won't stop."""
        problem = self._problem(FrrDownDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected FRR down to be verified: {result}")


class SDNControllerVerifyTest(unittest.TestCase):
    SCENARIO = "sdn_star"

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = CliRunner()

    def setUp(self) -> None:
        result = self.runner.invoke(app, ["env", "run", self.SCENARIO, "-t", "s"])
        if result.exit_code != 0:
            raise RuntimeError(f"nika env run failed:\n{result.output}")
        match = re.search(r"session_id=(\S+)", result.output.strip())
        if match is None:
            raise RuntimeError(f"session_id not found in env run output:\n{result.output}")
        self.session_id = match.group(1)
        row = SessionStore().get_session(self.session_id)
        self.lab_name = row["lab_name"]

    def tearDown(self) -> None:
        _teardown_env(self.runner, self.session_id)

    def _problem(self, cls_):
        return cls_(scenario_name=self.SCENARIO, lab_name=self.lab_name)

    def test_sdn_controller_crash_verify_true_after_inject(self):
        """verify_fault returns verified=True after POX controller is killed."""
        problem = self._problem(SDNControllerCrashDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected SDN controller crash to be verified: {result}")

    def test_southbound_port_block_verify_true_after_inject(self):
        """verify_fault returns verified=True after port 6633 is blocked via nftables."""
        problem = self._problem(SouthboundPortBlockDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected southbound port block to be verified: {result}")

    def test_southbound_port_mismatch_verify_true_after_inject(self):
        """verify_fault returns verified=True after POX is restarted on a mismatched port."""
        problem = self._problem(SouthboundPortMismatchDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected southbound port mismatch to be verified: {result}")

    def test_flow_rule_shadowing_verify_true_after_inject(self):
        """verify_fault returns verified=True after high-priority drop rule is added."""
        problem = self._problem(FlowRuleShadowingDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected flow rule shadowing to be verified: {result}")

    def test_flow_rule_loop_verify_true_after_inject(self):
        """verify_fault returns verified=True after loop rules are added to both switches."""
        problem = self._problem(FlowRuleLoopDetection)
        problem.inject_fault()
        result = problem.verify_fault()
        self.assertTrue(result["verified"], f"Expected flow rule loop to be verified: {result}")


if __name__ == "__main__":
    unittest.main()
