"""Unit tests for Containerlab topology parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nika.net_env.net_env_pool import get_net_env_instance
from nika.runtime.containerlab import parse_clab_topology


class ClabParseTest(unittest.TestCase):
    def test_parse_binds_and_exec(self) -> None:
        content = """
name: demo
topology:
  nodes:
    host:
      kind: linux
      image: alpine:latest
      binds:
        - /tmp/demo:/demo
      exec:
        - ip link set eth1 up
  links:
    - endpoints: ["host:eth1", "host:eth2"]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.clab.yml"
            path.write_text(content, encoding="utf-8")
            spec = parse_clab_topology(path)
            self.assertEqual(spec.nodes[0].binds, ["/tmp/demo:/demo"])
            self.assertEqual(spec.nodes[0].exec_cmds, ["ip link set eth1 up"])

    def test_invalid_topology_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.clab.yml"
            path.write_text("nodes: {}", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_clab_topology(path)

    def test_min3clos_template_parses(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "src/nika/net_env/containerlab/min3clos/min3clos.clab.yml.tmpl"
        )
        spec = parse_clab_topology(template)
        self.assertEqual(spec.name, "__LAB_NAME__")
        node_names = [node.name for node in spec.nodes]
        self.assertEqual(len(node_names), 5)
        self.assertIn("leaf1", node_names)
        self.assertIn("spine", node_names)
        self.assertIn("client2", node_names)
        self.assertEqual(len(spec.links), 4)

    def test_min3clos_env_uses_parsed_topology(self) -> None:
        env = get_net_env_instance("min3clos", backend="containerlab")
        spec = env.get_lab_spec()
        self.assertEqual(spec.name, "min3clos")
        self.assertEqual(len(spec.nodes), 5)
        self.assertIn("leaf1", env.get_info())


if __name__ == "__main__":
    unittest.main()
