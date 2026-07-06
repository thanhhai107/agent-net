"""Integration tests for Containerlab srlceos01 scenario.

Prerequisites:
  - Docker running
  - containerlab CLI on PATH
  - Nokia SR Linux and Arista cEOS images available locally
  - Run via: uv run python -m unittest tests/net_env_verify/test_clab_srlceos01_verify.py -v
"""

from __future__ import annotations

import shutil
import unittest

from nika.runtime.factory import resolve_backend, runtime_for_session

from tests.integration_base import PerTestEnvTestCase

SRL_NODE = "srl"
CEOS_NODE = "ceos"
SRL_INTF = "e1-1"
CEOS_INTF = "eth1"


@unittest.skipUnless(shutil.which("clab"), "containerlab not installed")
class ClabSrlCeos01VerifyIntegrationTest(PerTestEnvTestCase):
    SCENARIO = "srlceos01"
    ENV_RUN_ARGS = ["--backend", "containerlab"]

    def _runtime(self):
        return runtime_for_session(self._session_row(self.session_id))

    @staticmethod
    def _operstate(runtime, node: str, intf: str) -> str:
        return runtime.exec(node, f"cat /sys/class/net/{intf}/operstate").strip()

    @staticmethod
    def _link_is_up(runtime, node: str, intf: str) -> bool:
        return ClabSrlCeos01VerifyIntegrationTest._operstate(runtime, node, intf) == "up"

    @staticmethod
    def _link_is_down(runtime, node: str, intf: str) -> bool:
        return ClabSrlCeos01VerifyIntegrationTest._operstate(runtime, node, intf) in (
            "down",
            "lowerlayerdown",
        )

    def test_session_uses_containerlab_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "containerlab")
        self.assertIsNotNone(row.get("topology_file"))

    def test_nodes_deployed(self) -> None:
        nodes = self._runtime().list_nodes()
        self.assertIn(SRL_NODE, nodes)
        self.assertIn(CEOS_NODE, nodes)

    def test_interconnect_links_up(self) -> None:
        runtime = self._runtime()
        self.assertTrue(
            self._link_is_up(runtime, SRL_NODE, SRL_INTF),
            f"{SRL_NODE}:{SRL_INTF} operstate={self._operstate(runtime, SRL_NODE, SRL_INTF)!r}",
        )
        self.assertTrue(
            self._link_is_up(runtime, CEOS_NODE, CEOS_INTF),
            f"{CEOS_NODE}:{CEOS_INTF} operstate={self._operstate(runtime, CEOS_NODE, CEOS_INTF)!r}",
        )

    def test_link_state_propagates_to_peer(self) -> None:
        runtime = self._runtime()
        self.assertTrue(self._link_is_up(runtime, SRL_NODE, SRL_INTF))
        self.assertTrue(self._link_is_up(runtime, CEOS_NODE, CEOS_INTF))

        runtime.exec(SRL_NODE, f"ip link set {SRL_INTF} down")
        self.assertTrue(
            self._link_is_down(runtime, SRL_NODE, SRL_INTF),
            f"{SRL_NODE}:{SRL_INTF} operstate={self._operstate(runtime, SRL_NODE, SRL_INTF)!r}",
        )
        self.assertTrue(
            self._link_is_down(runtime, CEOS_NODE, CEOS_INTF),
            f"{CEOS_NODE}:{CEOS_INTF} operstate={self._operstate(runtime, CEOS_NODE, CEOS_INTF)!r}",
        )


if __name__ == "__main__":
    unittest.main()
