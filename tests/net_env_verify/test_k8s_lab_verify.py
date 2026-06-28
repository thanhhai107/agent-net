"""Unit and integration tests for the k8s_lab scenario.

Unit tests (no Docker required)
--------------------------------
- Node classification (FRR routers, k3s nodes, client host).
- Machine flags (bridged as2r1, privileged k3s nodes).

Integration tests (require Docker + ``rancher/k3s`` images)
------------------------------------------------------------
- Deploy k8s_lab via CLI and verify the session is running.

Prerequisites:
  - Unit tests: uv run python -m unittest tests/net_env_verify/test_k8s_lab_verify.py -v
  - Integration: Docker running + ``rancher/k3s`` pulled; remove ``@unittest.skip`` on
    ``K8sLabStartupVerifyTest`` first.
"""

from __future__ import annotations

import unittest

from nika.net_env.kubernetes.k8s_lab.lab import K8sFatTreeBGP

from tests.integration_base import PerTestEnvTestCase


class K8sLabUnitTest(unittest.TestCase):
    """Verify k8s_lab lab structure without Docker."""

    def test_has_frr_routers(self) -> None:
        """k8s_lab must expose its FRR routers through the base-class routers list."""
        inst = K8sFatTreeBGP()
        self.assertTrue(len(inst.routers) > 0, "Expected at least one FRR router")
        expected_routers = {
            "leaf_1_1", "leaf_1_2", "spine_1_1", "spine_1_2",
            "spine_2_1", "spine_2_2", "leaf_2_1", "leaf_2_2",
            "core_1_1", "core_1_2", "dc_exit", "as1r1", "as2r1",
        }
        self.assertEqual(set(inst.routers), expected_routers)

    def test_has_kubernetes_nodes(self) -> None:
        """k8s_lab must classify k3s machines into kubernetes_nodes."""
        inst = K8sFatTreeBGP()
        expected_k8s = {"controller", "worker1", "worker2", "worker3", "worker4", "worker5"}
        self.assertEqual(set(inst.kubernetes_nodes), expected_k8s)

    def test_has_client_host(self) -> None:
        """k8s_lab must have the client node classified as a host."""
        inst = K8sFatTreeBGP()
        self.assertIn("client", inst.hosts)

    def test_as2r1_is_bridged(self) -> None:
        """as2r1 must be bridged to provide internet connectivity."""
        inst = K8sFatTreeBGP()
        self.assertTrue(inst.lab.machines["as2r1"].is_bridged())

    def test_k3s_nodes_are_privileged(self) -> None:
        """k3s nodes must run in privileged mode."""
        inst = K8sFatTreeBGP()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_privileged(),
                f"Expected {node_name} to be privileged but it is not",
            )


@unittest.skip(
    "Requires Docker + rancher/k3s image. "
    "Remove @skip and run manually: "
    "uv run python -m unittest tests/net_env_verify/test_k8s_lab_verify.py -v"
)
class K8sLabStartupVerifyTest(PerTestEnvTestCase):
    """Deploy k8s_lab via CLI and verify the session is running."""

    SCENARIO = K8sFatTreeBGP.LAB_NAME

    def test_session_running_and_listed(self) -> None:
        """Session must be running and k8s_lab must appear in env list."""
        list_output = self._invoke_ok(["env", "list"])
        self.assertIn(K8sFatTreeBGP.LAB_NAME, list_output)


if __name__ == "__main__":
    unittest.main()
