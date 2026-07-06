"""Tests for Containerlab min5clos scenario verification.

Unit tests (no Docker)
----------------------
- ``verify_min5clos_lab`` signal logic via mocks.

Integration tests (Docker + containerlab + gnmic required)
----------------------------------------------------------
- Deploy via ``start_net_env`` (connectivity checked by ``verify_lab``).
- Assert session backend and key nodes are present.

Run via: uv run python -m unittest tests.net_env_verify.test_clab_min5clos_verify -v
"""

from __future__ import annotations

import shutil
import unittest
from unittest.mock import MagicMock

import docker

from nika.net_env.containerlab.min5clos.verify import (
    CLIENT1,
    CLIENT1_GATEWAY,
    CLIENT1_IP,
    CLIENT2_IP,
    CLIENT3_IP,
    EXPECTED_NODES,
    LEAF1,
    LEAF3_LOOPBACK,
    verify_min5clos_lab,
)
from nika.runtime.factory import resolve_backend, runtime_for_session
from tests.integration_base import SharedSessionTestCase


def _docker_available() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _min5clos_prerequisites() -> bool:
    return bool(shutil.which("clab") and shutil.which("gnmic") and _docker_available())


class Min5ClosLabVerifyUnitTest(unittest.TestCase):
    def _runtime(self, responses: dict[tuple[str, str], str], nodes: list[str] | None = None) -> MagicMock:
        runtime = MagicMock()
        runtime.list_nodes.return_value = nodes if nodes is not None else list(EXPECTED_NODES)

        def exec_side_effect(node: str, cmd: str, timeout: float = 10.0) -> str:
            return responses.get((node, cmd), "")

        runtime.exec.side_effect = exec_side_effect
        return runtime

    def _passing_responses(self) -> dict[tuple[str, str], str]:
        return {
            (CLIENT1, "cat /sys/class/net/eth1/operstate"): "up",
            (CLIENT1, "ip -4 -o addr show dev eth1"): f"inet {CLIENT1_IP}/31",
            (
                LEAF1,
                'sr_cli -c "show network-instance default protocols bgp neighbor"',
            ): "Peer : 10.0.0.1\nSession-state : established\nPeer : 10.0.0.3\nSession-state : established\n",
            (CLIENT1, f"ping -c 1 -W 2 {CLIENT1_GATEWAY}"): "1 received",
            (CLIENT1, f"ping -c 1 -W 2 {CLIENT2_IP}"): "1 received",
            (CLIENT1, f"ping -c 1 -W 2 {CLIENT3_IP}"): "1 received",
            (CLIENT1, f"ping -c 1 -W 2 {LEAF3_LOOPBACK}"): "1 received",
        }

    def test_verify_passes_when_all_checks_ok(self) -> None:
        runtime = self._runtime(self._passing_responses())
        result = verify_min5clos_lab(runtime, scenario_name="min5clos")
        self.assertTrue(result["verified"])
        self.assertTrue(all(result["checks"].values()))

    def test_verify_fails_when_bgp_not_established(self) -> None:
        responses = self._passing_responses()
        responses[(LEAF1, 'sr_cli -c "show network-instance default protocols bgp neighbor"')] = "Peer : idle\n"
        result = verify_min5clos_lab(self._runtime(responses), scenario_name="min5clos")
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["leaf1_bgp_neighbors"])

    def test_verify_fails_when_cross_pod_unreachable(self) -> None:
        responses = self._passing_responses()
        responses[(CLIENT1, f"ping -c 1 -W 2 {CLIENT3_IP}")] = "0 received"
        result = verify_min5clos_lab(self._runtime(responses), scenario_name="min5clos")
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["cross_pod_client_reachable"])

    def test_verify_fails_when_nodes_missing(self) -> None:
        runtime = self._runtime(self._passing_responses(), nodes=["client1", "leaf1"])
        result = verify_min5clos_lab(runtime, scenario_name="min5clos")
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["nodes_deployed"])


@unittest.skipUnless(_min5clos_prerequisites(), "containerlab, gnmic, or Docker not available")
class Min5ClosVerifyIntegrationTest(SharedSessionTestCase):
    SCENARIO = "min5clos"
    ENV_RUN_ARGS = ["--backend", "containerlab"]

    def _runtime(self):
        return runtime_for_session(self._session_row(self.session_id))

    def test_session_uses_containerlab_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "containerlab")
        self.assertIn(self.SCENARIO, row["lab_name"])
        self.assertIsNotNone(row.get("topology_file"))

    def test_all_nodes_deployed(self) -> None:
        nodes = set(self._runtime().list_nodes())
        for name in EXPECTED_NODES:
            self.assertIn(name, nodes, f"Expected node {name!r} in deployed lab")

    def test_cross_pod_ping_from_client1(self) -> None:
        runtime = self._runtime()
        output = runtime.exec(CLIENT1, f"ping -c 1 -W 2 {CLIENT3_IP}", timeout=10)
        self.assertIn("1 received", output, f"client1 -> client3 ping failed: {output!r}")


if __name__ == "__main__":
    unittest.main()
