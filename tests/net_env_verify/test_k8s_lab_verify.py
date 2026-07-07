"""Unit and integration tests for the k8s_lab scenario.

Unit tests (no Docker required)
--------------------------------
- Node classification (FRR routers, k3s nodes, client host).
- Machine flags (bridged as2r1, privileged k3s nodes).

Integration tests (require Docker, root, and pre-pulled images)
---------------------------------------------------------------
- Deploy k8s_lab and verify BGP fabric, k3s cluster, ingress, and client access
  using ``nika exec`` / Kathara exec (same command path as the CLI).

Prerequisites:
  - Unit tests: uv run python -m unittest tests/net_env_verify/test_k8s_lab_verify.py -v
  - Integration: Docker running as root (k3s nodes are privileged), images pulled via
    ``ensure_k8s_lab_images()`` on first deploy.
"""

from __future__ import annotations

import os
import time
import unittest

import docker

from nika.net_env.kathara.kubernetes.k8s_lab.lab import K8sFatTreeBGP
from nika.service.kathara.base_api import KatharaBaseAPI

from tests.integration_base import SharedSessionTestCase


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _privileged_lab_supported() -> bool:
    """k3s machines require privileged containers; Kathara needs root for that."""
    return os.geteuid() == 0


class K8sLabUnitTest(unittest.TestCase):
    """Verify k8s_lab lab structure without Docker."""

    def test_has_frr_routers(self) -> None:
        """k8s_lab must expose its FRR routers through the base-class routers list."""
        inst = K8sFatTreeBGP()
        self.assertTrue(len(inst.routers) > 0, "Expected at least one FRR router")
        expected_routers = {
            "leaf_1_1",
            "leaf_1_2",
            "spine_1_1",
            "spine_1_2",
            "spine_2_1",
            "spine_2_2",
            "leaf_2_1",
            "leaf_2_2",
            "core_1_1",
            "core_1_2",
            "dc_exit",
            "as1r1",
            "as2r1",
        }
        self.assertEqual(set(inst.routers), expected_routers)

    def test_has_kubernetes_nodes(self) -> None:
        """k8s_lab must classify k3s machines into kubernetes_nodes."""
        inst = K8sFatTreeBGP()
        expected_k8s = {
            "controller",
            "worker1",
            "worker2",
            "worker3",
            "worker4",
            "worker5",
        }
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


@unittest.skipUnless(
    _docker_available() and _privileged_lab_supported(),
    "Requires Docker and root (privileged k3s containers)",
)
class K8sLabIntegrationTest(SharedSessionTestCase):
    """End-to-end checks for k8s_lab after deploy and controller.startup."""

    SCENARIO = K8sFatTreeBGP.LAB_NAME
    _READY_TIMEOUT_SEC = 900

    _api: KatharaBaseAPI

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._api = KatharaBaseAPI(lab_name=cls._lab_name())
        cls._wait_until_ready()

    @classmethod
    def _lab_name(cls) -> str:
        from nika.utils.session_store import SessionStore

        return SessionStore().get_session(cls.session_id)["lab_name"]

    @classmethod
    def _exec(cls, host: str, command: str, timeout: float = 120) -> str:
        return cls._api.exec_cmd(host, command, timeout=timeout)

    @classmethod
    def _wait_until_ready(cls) -> None:
        """Poll until k3s, ingress, and sample apps are serving traffic."""
        deadline = time.time() + cls._READY_TIMEOUT_SEC
        last_error = "timeout"
        while time.time() < deadline:
            try:
                nodes = cls._exec(
                    "controller", "kubectl get nodes --no-headers", timeout=60
                )
                ready_nodes = [
                    line
                    for line in nodes.splitlines()
                    if line.strip().endswith(" Ready")
                ]
                if len(ready_nodes) < 6:
                    last_error = f"k3s nodes not ready ({len(ready_nodes)}/6)"
                    time.sleep(15)
                    continue

                ingress = cls._exec(
                    "controller",
                    "kubectl get svc -n ingress-nginx ingress-nginx-controller "
                    "-o jsonpath={.status.loadBalancer.ingress[0].ip}",
                    timeout=60,
                ).strip()
                if not ingress.startswith("101."):
                    last_error = f"ingress VIP missing (got {ingress!r})"
                    time.sleep(15)
                    continue

                code = cls._exec(
                    "client",
                    "curl -s -o /dev/null -w '%{http_code}' http://datacenter.com/word",
                    timeout=60,
                ).strip()
                if code != "200":
                    last_error = f"word app HTTP {code!r}"
                    time.sleep(15)
                    continue
                return
            except Exception as exc:  # noqa: BLE001 - poll until deadline
                last_error = str(exc)
                time.sleep(15)
        raise TimeoutError(
            f"k8s_lab not ready within {cls._READY_TIMEOUT_SEC}s: {last_error}"
        )

    def test_bgp_spine_neighbors_up(self) -> None:
        """Pod-1 leaf routers must peer with spine routers (AS 64514)."""
        output = self._exec("leaf_1_1", "vtysh -c 'show bgp summary'")
        self.assertIn("64514", output)
        self.assertRegex(output, r"eth[01]\s+4\s+64514\s+\d+\s+\d+\s+\d+")

    def test_metallb_route_on_leaf(self) -> None:
        """MetalLB VIP must be reachable in the leaf routing table via BGP."""
        output = self._exec("leaf_1_1", "vtysh -c 'show ip route'")
        self.assertRegex(output, r"101\.0\.0\.1/32")

    def test_k3s_cluster_ready(self) -> None:
        """All six k3s nodes must report Ready."""
        output = self._exec("controller", "kubectl get nodes --no-headers")
        ready = [
            line for line in output.splitlines() if line.strip().endswith(" Ready")
        ]
        self.assertEqual(len(ready), 6, output)

    def test_ingress_loadbalancer_vip(self) -> None:
        """Ingress controller must receive a MetalLB IP in 101.0.0.0/8."""
        output = self._exec(
            "controller",
            "kubectl get svc -n ingress-nginx ingress-nginx-controller",
        )
        self.assertRegex(output, r"101\.\d+\.\d+\.\d+")

    def test_cross_leaf_reachability(self) -> None:
        """Controller (leaf_1_1) must reach worker3 (leaf_1_2)."""
        output = self._exec("controller", "ping -c 3 201.2.1.2")
        self.assertIn("3 packets received", output)

    def test_client_reaches_controller(self) -> None:
        """External client must reach the k3s controller host IP."""
        output = self._exec("client", "ping -c 3 201.1.1.2")
        self.assertIn("3 received", output)

    def test_client_word_app_http(self) -> None:
        """Client must reach the word app through ingress."""
        code = self._exec(
            "client",
            "curl -s -o /dev/null -w '%{http_code}' http://datacenter.com/word",
        ).strip()
        self.assertEqual(code, "200")

    def test_client_weather_app_http(self) -> None:
        """Client must reach the weather app through ingress."""
        code = self._exec(
            "client",
            "curl -s -o /dev/null -w '%{http_code}' 'http://datacenter.com/weather?location=London'",
        ).strip()
        self.assertEqual(code, "200")
        body = self._exec(
            "client",
            "curl -s 'http://datacenter.com/weather?location=London'",
        )
        self.assertIn("London", body)


if __name__ == "__main__":
    unittest.main()
