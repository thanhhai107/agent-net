"""Unit tests for KatharaRuntime and factory defaults."""

from __future__ import annotations

import unittest

from nika.runtime.factory import resolve_backend, runtime_for_net_env
from nika.runtime.kathara import KatharaRuntime
from nika.net_env.kathara.interdomain_routing.simple_bgp.lab import SimpleBGP


class KatharaRuntimeCompatTest(unittest.TestCase):
    def test_factory_default_backend(self) -> None:
        self.assertEqual(resolve_backend({}), "kathara")

    def test_runtime_for_kathara_net_env(self) -> None:
        env = SimpleBGP()
        runtime = runtime_for_net_env(env)
        self.assertIsInstance(runtime, KatharaRuntime)
        self.assertEqual(runtime.lab_name, env.name)

    def test_kathara_runtime_list_nodes_before_deploy(self) -> None:
        env = SimpleBGP()
        runtime = KatharaRuntime(env)
        nodes = runtime.list_nodes()
        self.assertIn("pc1", nodes)
        self.assertIn("router1", nodes)


if __name__ == "__main__":
    unittest.main()
