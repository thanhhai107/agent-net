"""Unit tests for LabRuntime semantic operations."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from nika.runtime.base import LabRuntime
from nika.runtime import ops_defaults


class _StubRuntime(LabRuntime):
    def __init__(self, responses: dict[tuple[str, str], str] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    @property
    def lab_name(self) -> str:
        return "stub_lab"

    def deploy(self) -> None:
        pass

    def destroy(self) -> None:
        pass

    def exists(self) -> bool:
        return True

    def inspect(self) -> list[dict]:
        return []

    def list_nodes(self) -> list[str]:
        return ["pc1", "client1", "router1"]

    def exec(self, node: str, cmd: str, *, timeout: float = 10.0) -> str:
        self.calls.append((node, cmd))
        return self._responses.get((node, cmd), "")

    def get_container(self, node: str):
        container = MagicMock()
        container.status = self._responses.get(("__status__", node), "running")
        container.reload = MagicMock()
        return container

    def pause(self, node: str) -> None:
        pass

    def unpause(self, node: str) -> None:
        pass


class LabOpsTest(unittest.TestCase):
    def test_get_interface_operstate(self):
        runtime = _StubRuntime({("pc1", "cat /sys/class/net/eth0/operstate"): "down\n"})
        self.assertEqual(runtime.get_interface_operstate("pc1", "eth0"), "down")

    def test_get_host_ip_prefers_interface(self):
        addr_json = json.dumps(
            [
                {
                    "ifname": "eth0",
                    "addr_info": [{"family": "inet", "local": "10.0.0.2", "prefixlen": 24}],
                }
            ]
        )
        runtime = _StubRuntime({("pc1", "ip -j addr"): addr_json})
        self.assertEqual(runtime.get_host_ip("pc1", "eth0", with_prefix=True), "10.0.0.2/24")

    def test_get_default_gateway(self):
        route_json = json.dumps([{"dst": "default", "gateway": "10.0.0.1"}])
        runtime = _StubRuntime({("pc1", "ip -j route"): route_json})
        self.assertEqual(runtime.get_default_gateway("pc1"), "10.0.0.1")

    def test_add_nft_drop_rule_builds_commands(self):
        runtime = _StubRuntime()
        runtime.add_nft_drop_rule("router1", "tcp dport 179 drop")
        cmds = [cmd for _, cmd in runtime.calls]
        self.assertTrue(any("nft add table inet filter" in cmd for cmd in cmds))
        self.assertTrue(any("tcp dport 179 drop" in cmd for cmd in cmds))

    def test_node_status_paused(self):
        runtime = _StubRuntime({("__status__", "pc1"): "paused"})
        self.assertEqual(runtime.node_status("pc1"), "paused")

    def test_list_dhcp_client_nodes(self):
        runtime = _StubRuntime()
        self.assertEqual(runtime.list_dhcp_client_nodes(), ["pc1", "client1"])

    def test_dhcp_set_option_routers(self):
        runtime = _StubRuntime()
        runtime.dhcp_set_option_routers("dhcp1", "192.168.1.0", "192.168.1.254")
        cmds = [cmd for _, cmd in runtime.calls]
        self.assertTrue(any("option routers 192.168.1.254" in cmd for cmd in cmds))
        self.assertTrue(any("systemctl restart isc-dhcp-server" in cmd for cmd in cmds))

    def test_tc_set_netem_command(self):
        runtime = _StubRuntime()
        runtime.tc_set_netem("pc1", "eth0", corrupt=60)
        self.assertIn("corrupt 60%", runtime.calls[0][1])

    def test_write_file_uses_base64(self):
        runtime = _StubRuntime()
        ops_defaults.write_file(runtime, "pc1", "/tmp/x.txt", "hello")
        self.assertIn("base64 -d", runtime.calls[0][1])


if __name__ == "__main__":
    unittest.main()
