"""Integration tests for Containerlab min3clos scenario.

Prerequisites:
  - Docker running
  - containerlab CLI on PATH
  - gnmic installed
  - ghcr.io/nokia/srlinux:24.10 and ghcr.io/hellt/network-multitool images

Deploy via ``start_net_env`` (``verify_lab`` runs during startup). Run via:
  uv run python -m unittest tests.nika.net_env.test_clab_min3clos_verify -v
"""

from __future__ import annotations

import unittest
from typing import ClassVar

from nika.net_env.containerlab.min3clos.verify import (
    CLIENT1,
    CLIENT2_IP,
    EXPECTED_NODES,
)
from nika.runtime.factory import resolve_backend, runtime_for_session
from tests.support.integration_base import SharedSessionTestCase
from tests.support.prerequisites import containerlab_prerequisites


@unittest.skipUnless(
    containerlab_prerequisites(), "containerlab, gnmic, or Docker not available"
)
class Min3ClosVerifyIntegrationTest(SharedSessionTestCase):
    SCENARIO = "min3clos"
    ENV_RUN_ARGS: ClassVar[list[str]] = []

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

    def test_cross_leaf_ping_from_client1(self) -> None:
        runtime = self._runtime()
        output = runtime.exec(CLIENT1, f"ping -c 1 -W 2 {CLIENT2_IP}", timeout=10)
        self.assertIn(
            "1 received", output, f"client1 -> client2 ping failed: {output!r}"
        )


if __name__ == "__main__":
    unittest.main()
