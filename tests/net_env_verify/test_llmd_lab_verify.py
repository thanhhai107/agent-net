"""Unit and integration tests for the llmd_lab scenario.

Unit tests (no Docker required)
--------------------------------
- Node classification (k3s nodes, client host).
- Machine flags (bridged k3s nodes, privileged k3s nodes).

Integration tests (require Docker + ``rancher/k3s`` images)
------------------------------------------------------------
- Deploy llmd_lab via CLI and verify the session is running.

Prerequisites:
  - Unit tests: uv run python -m unittest tests/net_env_verify/test_llmd_lab_verify.py -v
  - Integration: Docker running + ``rancher/k3s`` pulled; remove ``@unittest.skip`` on
    ``LLMDLabStartupVerifyTest`` first.
"""

from __future__ import annotations

import unittest

from nika.net_env.kubernetes.llmd_lab.lab import LLMDInferenceCluster

from tests.integration_base import PerTestEnvTestCase


class LLMDLabUnitTest(unittest.TestCase):
    """Verify llmd_lab lab structure without Docker."""

    def test_has_kubernetes_nodes(self) -> None:
        """llmd_lab must classify all k3s machines into kubernetes_nodes."""
        inst = LLMDInferenceCluster()
        expected_k8s = {"controller", "worker1", "worker2", "worker3", "worker4", "worker5"}
        self.assertEqual(set(inst.kubernetes_nodes), expected_k8s)

    def test_has_client_host(self) -> None:
        """llmd_lab must have the client node classified as a host."""
        inst = LLMDInferenceCluster()
        self.assertIn("client", inst.hosts)

    def test_all_k3s_nodes_are_bridged(self) -> None:
        """All k3s nodes must have bridged=True for internet access."""
        inst = LLMDInferenceCluster()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_bridged(),
                f"Expected {node_name} to be bridged but it is not",
            )

    def test_k3s_nodes_are_privileged(self) -> None:
        """k3s nodes must run in privileged mode."""
        inst = LLMDInferenceCluster()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_privileged(),
                f"Expected {node_name} to be privileged but it is not",
            )


@unittest.skip(
    "Requires Docker + rancher/k3s image. "
    "Remove @skip and run manually: "
    "uv run python -m unittest tests/net_env_verify/test_llmd_lab_verify.py -v"
)
class LLMDLabStartupVerifyTest(PerTestEnvTestCase):
    """Deploy llmd_lab via CLI and verify the session is running."""

    SCENARIO = LLMDInferenceCluster.LAB_NAME

    def test_session_running_and_listed(self) -> None:
        """Session must be running and llmd_lab must appear in env list."""
        list_output = self._invoke_ok(["env", "list"])
        self.assertIn(LLMDInferenceCluster.LAB_NAME, list_output)


if __name__ == "__main__":
    unittest.main()
