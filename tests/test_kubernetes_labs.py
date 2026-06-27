"""Tests for Kubernetes labs integration into the net_env module.

This file contains two test tiers:

Unit tests (no Docker required)
--------------------------------
- Registration: both labs appear in the pool and can be looked up by name.
- Discovery: ``list_all_net_envs()`` and ``nika env list`` CLI include both labs.
- Instantiation: constructors build valid Kathara Lab objects.

Integration tests (require Docker + ``rancher/k3s`` images)
------------------------------------------------------------
- ``K8sLabStartupIntegrationTest`` / ``LLMDLabStartupIntegrationTest``
  follow the same ``PerTestEnvTestCase`` pattern as the existing
  failure-inject verify tests.  Run only if Docker is available and
  the k3s image has been pulled.

Failure injection
-----------------
Traditional failure injection (link-down, flap, etc.) targets network
interfaces inside Kathara containers.  The Kubernetes labs do expose
suitable containers:

* **k8s_lab** – the ``client`` node uses ``kathara/base`` and has an
  eth0 interface into the BGP network; FRR router containers are also
  present.  Link injection on these nodes is *technically* supported by
  the existing pipeline.

* **llmd_lab** – the ``client`` node uses ``kathara/base`` and is
  connected to the star link.  Link injection on the client is
  technically supported.

However, because both labs require k3s to complete its full
initialisation sequence (which pulls container images over the
internet and can take 20-40 minutes), running a fully automated
deployment + injection test is **not practical** in a standard CI
environment.  No failure-injection test is included here to avoid
false results.  When k3s images are locally cached and Docker is
available, the standard ``PerTestEnvTestCase``-based integration tests
below will exercise the startup pipeline end-to-end and can be
extended with failure injection once the environment is confirmed
stable.
"""

from __future__ import annotations

import unittest

from typer.testing import CliRunner

from nika.codex_cli.main import app
from nika.net_env.kubernetes.k8s_lab.lab import K8sFatTreeBGP
from nika.net_env.kubernetes.llmd_lab.lab import LLMDInferenceCluster
from nika.net_env.net_env_pool import get_net_env_instance, list_all_net_envs


# ---------------------------------------------------------------------------
# Registration & discovery  (no Docker required)
# ---------------------------------------------------------------------------


class TestKubernetesLabsRegistration(unittest.TestCase):
    """Verify both kubernetes labs are registered and discoverable."""

    def test_k8s_lab_in_pool(self) -> None:
        """k8s_lab must be present in the global net-env pool."""
        self.assertIn(K8sFatTreeBGP.LAB_NAME, list_all_net_envs())

    def test_llmd_lab_in_pool(self) -> None:
        """llmd_lab must be present in the global net-env pool."""
        self.assertIn(LLMDInferenceCluster.LAB_NAME, list_all_net_envs())

    def test_env_list_cli_includes_k8s_lab(self) -> None:
        """``nika env list`` output must include k8s_lab."""
        runner = CliRunner()
        result = runner.invoke(app, ["env", "list"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(K8sFatTreeBGP.LAB_NAME, result.output)

    def test_env_list_cli_includes_llmd_lab(self) -> None:
        """``nika env list`` output must include llmd_lab."""
        runner = CliRunner()
        result = runner.invoke(app, ["env", "list"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(LLMDInferenceCluster.LAB_NAME, result.output)


# ---------------------------------------------------------------------------
# Instantiation  (no Docker required)
# ---------------------------------------------------------------------------


class TestKubernetesLabsInstantiation(unittest.TestCase):
    """Verify that lab constructors build valid Kathara Lab objects."""

    # K8sFatTreeBGP ---------------------------------------------------------

    def test_k8s_lab_instance_via_pool(self) -> None:
        """get_net_env_instance('k8s_lab') returns a correctly initialised object."""
        inst = get_net_env_instance(K8sFatTreeBGP.LAB_NAME)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.name, K8sFatTreeBGP.LAB_NAME)
        self.assertIsInstance(inst, K8sFatTreeBGP)

    def test_k8s_lab_has_frr_routers(self) -> None:
        """k8s_lab must expose its FRR routers through the base-class routers list."""
        inst = K8sFatTreeBGP()
        self.assertTrue(len(inst.routers) > 0, "Expected at least one FRR router")
        expected_routers = {
            "leaf_1_1", "leaf_1_2", "spine_1_1", "spine_1_2",
            "spine_2_1", "spine_2_2", "leaf_2_1", "leaf_2_2",
            "core_1_1", "core_1_2", "dc_exit", "as1r1", "as2r1",
        }
        self.assertEqual(set(inst.routers), expected_routers)

    def test_k8s_lab_has_kubernetes_nodes(self) -> None:
        """k8s_lab must classify k3s machines into kubernetes_nodes."""
        inst = K8sFatTreeBGP()
        expected_k8s = {"controller", "worker1", "worker2", "worker3", "worker4", "worker5"}
        self.assertEqual(set(inst.kubernetes_nodes), expected_k8s)

    def test_k8s_lab_has_client_host(self) -> None:
        """k8s_lab must have the client node classified as a host."""
        inst = K8sFatTreeBGP()
        self.assertIn("client", inst.hosts)

    def test_k8s_lab_lab_name_constant(self) -> None:
        self.assertEqual(K8sFatTreeBGP.LAB_NAME, "k8s_lab")

    # LLMDInferenceCluster --------------------------------------------------

    def test_llmd_lab_instance_via_pool(self) -> None:
        """get_net_env_instance('llmd_lab') returns a correctly initialised object."""
        inst = get_net_env_instance(LLMDInferenceCluster.LAB_NAME)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.name, LLMDInferenceCluster.LAB_NAME)
        self.assertIsInstance(inst, LLMDInferenceCluster)

    def test_llmd_lab_has_kubernetes_nodes(self) -> None:
        """llmd_lab must classify all k3s machines into kubernetes_nodes."""
        inst = LLMDInferenceCluster()
        expected_k8s = {"controller", "worker1", "worker2", "worker3", "worker4", "worker5"}
        self.assertEqual(set(inst.kubernetes_nodes), expected_k8s)

    def test_llmd_lab_has_client_host(self) -> None:
        """llmd_lab must have the client node classified as a host."""
        inst = LLMDInferenceCluster()
        self.assertIn("client", inst.hosts)

    def test_llmd_lab_lab_name_constant(self) -> None:
        self.assertEqual(LLMDInferenceCluster.LAB_NAME, "llmd_lab")

    def test_llmd_lab_all_k3s_nodes_are_bridged(self) -> None:
        """All k3s nodes in llmd_lab must have bridged=True for internet access."""
        inst = LLMDInferenceCluster()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_bridged(),
                f"Expected {node_name} to be bridged but it is not",
            )

    def test_k8s_lab_as2r1_is_bridged(self) -> None:
        """as2r1 in k8s_lab must be bridged to provide internet connectivity."""
        inst = K8sFatTreeBGP()
        self.assertTrue(inst.lab.machines["as2r1"].is_bridged())

    def test_k8s_lab_k3s_nodes_are_privileged(self) -> None:
        """k3s nodes must run in privileged mode."""
        inst = K8sFatTreeBGP()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_privileged(),
                f"Expected {node_name} to be privileged but it is not",
            )

    def test_llmd_lab_k3s_nodes_are_privileged(self) -> None:
        """k3s nodes in llmd_lab must run in privileged mode."""
        inst = LLMDInferenceCluster()
        for node_name in inst.kubernetes_nodes:
            machine = inst.lab.machines[node_name]
            self.assertTrue(
                machine.is_privileged(),
                f"Expected {node_name} to be privileged but it is not",
            )


# ---------------------------------------------------------------------------
# Integration tests – require Docker + rancher/k3s images
# ---------------------------------------------------------------------------
# These tests follow the same PerTestEnvTestCase pattern used throughout the
# project (e.g. tests/failure_inject_verify/).  They deploy a real Kathara
# lab and therefore require:
#   1. Docker running
#   2. ``rancher/k3s`` image pulled locally
#   3. Sufficient CPU / RAM for a 7-node k3s cluster
#
# Due to these heavy prerequisites they are disabled by default.
# To run them:
#   uv run python -m unittest tests.test_kubernetes_labs.K8sLabStartupIntegrationTest -v
#   uv run python -m unittest tests.test_kubernetes_labs.LLMDLabStartupIntegrationTest -v
#
# NOTE ON FAILURE INJECTION
# --------------------------
# Both labs expose suitable containers for traditional link-failure injection
# (the ``client`` node and, in k8s_lab, all FRR router nodes).  However,
# because k3s startup can take 20-40 minutes (image pulls included),
# automated failure-injection tests are not included here to avoid
# unreliable CI results.  Once k3s images are locally cached the integration
# tests below can be extended with failure injection following the pattern in
# tests/failure_inject_verify/test_link_failure_verify.py.
# ---------------------------------------------------------------------------


@unittest.skip(
    "Requires Docker + rancher/k3s image. "
    "Remove @skip and run manually: "
    "uv run python -m unittest tests.test_kubernetes_labs.K8sLabStartupIntegrationTest -v"
)
class K8sLabStartupIntegrationTest(unittest.TestCase):
    """Deploy k8s_lab via CLI and verify the session is running.

    Mirrors the pattern of tests/test_pipeline.py step_01/step_02.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from typer.testing import CliRunner as _Runner

        cls.runner = _Runner()

    def _invoke_ok(self, args: list[str]) -> str:
        result = self.runner.invoke(app, args)
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output

    def test_k8s_lab_start_and_list(self) -> None:
        import re

        from nika.utils.session_store import SessionStore
        from nika.workflows.eval.clean import remove_session_results

        run_output = self._invoke_ok(["env", "run", K8sFatTreeBGP.LAB_NAME])
        match = re.search(r"session_id=(\S+)", run_output)
        self.assertIsNotNone(match, f"session_id missing:\n{run_output}")
        session_id = match.group(1)

        try:
            row = SessionStore().get_session(session_id)
            self.assertEqual(row["status"], "running")
            self.assertEqual(row["scenario_name"], K8sFatTreeBGP.LAB_NAME)

            list_output = self._invoke_ok(["env", "list"])
            self.assertIn(K8sFatTreeBGP.LAB_NAME, list_output)
        finally:
            self.runner.invoke(app, ["session", "close", session_id, "-y"])
            remove_session_results(session_id)


@unittest.skip(
    "Requires Docker + rancher/k3s image. "
    "Remove @skip and run manually: "
    "uv run python -m unittest tests.test_kubernetes_labs.LLMDLabStartupIntegrationTest -v"
)
class LLMDLabStartupIntegrationTest(unittest.TestCase):
    """Deploy llmd_lab via CLI and verify the session is running."""

    @classmethod
    def setUpClass(cls) -> None:
        from typer.testing import CliRunner as _Runner

        cls.runner = _Runner()

    def _invoke_ok(self, args: list[str]) -> str:
        result = self.runner.invoke(app, args)
        self.assertEqual(result.exit_code, 0, result.output)
        return result.output

    def test_llmd_lab_start_and_list(self) -> None:
        import re

        from nika.utils.session_store import SessionStore
        from nika.workflows.eval.clean import remove_session_results

        run_output = self._invoke_ok(["env", "run", LLMDInferenceCluster.LAB_NAME])
        match = re.search(r"session_id=(\S+)", run_output)
        self.assertIsNotNone(match, f"session_id missing:\n{run_output}")
        session_id = match.group(1)

        try:
            row = SessionStore().get_session(session_id)
            self.assertEqual(row["status"], "running")
            self.assertEqual(row["scenario_name"], LLMDInferenceCluster.LAB_NAME)

            list_output = self._invoke_ok(["env", "list"])
            self.assertIn(LLMDInferenceCluster.LAB_NAME, list_output)
        finally:
            self.runner.invoke(app, ["session", "close", session_id, "-y"])
            remove_session_results(session_id)


if __name__ == "__main__":
    unittest.main()
