"""Unit tests for Containerlab topology parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nika.net_env.net_env_pool import get_net_env_instance
from nika.runtime.containerlab import parse_clab_topology

_SRLCEOS01_TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src/nika/net_env/containerlab/srlceos01/srlceos01.clab.yml.tmpl"
)


class ClabParseTest(unittest.TestCase):
    def test_parse_srlceos01_template(self) -> None:
        spec = parse_clab_topology(_SRLCEOS01_TEMPLATE)
        self.assertEqual(spec.name, "__LAB_NAME__")
        self.assertEqual([node.name for node in spec.nodes], ["srl", "ceos"])
        self.assertEqual(spec.nodes[0].image, "ghcr.io/nokia/srlinux:24.10")
        self.assertEqual(spec.nodes[0].kind, "nokia_srlinux")
        self.assertEqual(spec.nodes[1].image, "ceos:4.32.0F")
        self.assertEqual(spec.nodes[1].kind, "arista_ceos")
        self.assertEqual(len(spec.links), 1)
        self.assertEqual(spec.links[0].endpoints, ("srl:ethernet-1/1", "ceos:eth1"))

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

    def test_srlceos01_env_uses_parsed_topology(self) -> None:
        env = get_net_env_instance("srlceos01", backend="containerlab")
        self.assertEqual(env.get_topology(), [("srl:ethernet-1/1", "ceos:eth1")])
        spec = env.get_lab_spec()
        self.assertEqual(spec.name, "srlceos01")
        self.assertIn("Nodes: srl, ceos", env.get_info())

    def test_min5clos_template_parses(self) -> None:
        template = (
            Path(__file__).resolve().parents[2]
            / "src/nika/net_env/containerlab/min5clos/min5clos.clab.yml.tmpl"
        )
        spec = parse_clab_topology(template)
        self.assertEqual(spec.name, "__LAB_NAME__")
        node_names = [node.name for node in spec.nodes]
        self.assertEqual(len(node_names), 14)
        self.assertIn("leaf1", node_names)
        self.assertIn("superspine2", node_names)
        self.assertIn("client4", node_names)
        self.assertEqual(len(spec.links), 16)

    def test_min5clos_env_uses_parsed_topology(self) -> None:
        env = get_net_env_instance("min5clos", backend="containerlab")
        spec = env.get_lab_spec()
        self.assertEqual(spec.name, "min5clos")
        self.assertEqual(len(spec.nodes), 14)
        self.assertIn("leaf1", env.get_info())


if __name__ == "__main__":
    unittest.main()
