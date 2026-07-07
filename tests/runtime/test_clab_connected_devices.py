"""Unit tests for Containerlab topology neighbor resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from nika.runtime.containerlab.runtime import ContainerlabRuntime


class ClabConnectedDevicesTest(unittest.TestCase):
    def test_min3clos_neighbors(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "src/nika/net_env/containerlab/min3clos/min3clos.clab.yml.tmpl"
        )
        runtime = ContainerlabRuntime(
            lab_name="min3clos__test",
            topology_file=template,
        )
        self.assertEqual(runtime.get_connected_devices("leaf1"), ["client1", "spine"])
        self.assertEqual(runtime.get_connected_devices("client1"), ["leaf1"])


if __name__ == "__main__":
    unittest.main()
