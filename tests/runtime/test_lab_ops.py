"""Unit tests for LabRuntime semantic operations."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from nika.runtime.base import LabRuntime, RuntimeCapabilityError
from nika.runtime import ops_defaults
from nika.orchestrator.problems.problem_base import ProblemBase


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


class _LimitedRuntime(_StubRuntime):
    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"exec"})


class _CapabilityProblem(ProblemBase):
    root_cause_name = "capability_problem"
    required_capabilities = ("tc",)


class LabOpsTest(unittest.TestCase):
    def test_has_capability(self):
        runtime = _StubRuntime()
        self.assertTrue(runtime.has_capability("exec"))
        self.assertTrue(runtime.has_capability("tc"))
        self.assertFalse(runtime.has_capability("unsupported"))

    def test_require_capabilities_passes_for_supported_capabilities(self):
        runtime = _StubRuntime()
        runtime.require_capabilities("exec", "interface")

    def test_require_capabilities_raises_clear_error_for_missing_capability(self):
        runtime = _LimitedRuntime()
        with self.assertRaisesRegex(
            RuntimeCapabilityError, "tc.*Supported capabilities: exec"
        ):
            runtime.require_capabilities("tc")

    def test_semantic_operation_raises_clear_error_for_missing_capability(self):
        runtime = _LimitedRuntime()
        with self.assertRaisesRegex(RuntimeCapabilityError, "does not support.*tc"):
            runtime.tc_show_intf("pc1", "eth0")

    def test_problem_runtime_capability_check_uses_runtime_declaration(self):
        problem = _CapabilityProblem()
        problem.runtime = _LimitedRuntime()
        problem.net_env = object()
        with self.assertRaisesRegex(
            RuntimeCapabilityError, "capability_problem|_CapabilityProblem.*tc"
        ):
            problem.check_runtime_compatible(operation="inject_fault")

    def test_get_interface_operstate(self):
        runtime = _StubRuntime({("pc1", "cat /sys/class/net/eth0/operstate"): "down\n"})
        self.assertEqual(runtime.get_interface_operstate("pc1", "eth0"), "down")

    def test_get_host_ip_prefers_interface(self):
        addr_json = json.dumps(
            [
                {
                    "ifname": "eth0",
                    "addr_info": [
                        {"family": "inet", "local": "10.0.0.2", "prefixlen": 24}
                    ],
                }
            ]
        )
        runtime = _StubRuntime({("pc1", "ip -j addr"): addr_json})
        self.assertEqual(
            runtime.get_host_ip("pc1", "eth0", with_prefix=True), "10.0.0.2/24"
        )

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

    def test_tc_set_tbf_accepts_host_name_alias(self):
        runtime = _StubRuntime()
        runtime.tc_set_tbf(
            host_name="pc1", intf_name="eth0", rate="1mbit", burst="64kb", limit="500kb"
        )
        self.assertIn("tbf rate 1mbit", runtime.calls[0][1])

    def test_write_file_uses_base64(self):
        runtime = _StubRuntime()
        ops_defaults.write_file(runtime, "pc1", "/tmp/x.txt", "hello")
        self.assertIn("base64 -d", runtime.calls[0][1])

    def test_frr_get_bgp_asn_number_from_summary(self):
        runtime = _StubRuntime(
            {
                (
                    "router1",
                    "vtysh -c 'show bgp summary' 2>/dev/null || true",
                ): "BGP router identifier 10.0.0.1, local AS number 65001 vrf-id 0\n"
            }
        )
        self.assertEqual(runtime.frr_get_bgp_asn_number("router1"), 65001)

    def test_frr_get_bgp_asn_number_falls_back_to_running_config(self):
        runtime = _StubRuntime(
            {
                ("router1", "vtysh -c 'show bgp summary' 2>/dev/null || true"): "",
                (
                    "router1",
                    "vtysh -c 'show running-config' 2>/dev/null | grep -E '^router bgp ' | awk '{print $3}' | head -n1",
                ): "2\n",
            }
        )
        self.assertEqual(runtime.frr_get_bgp_asn_number("router1"), 2)

    def test_process_running(self):
        runtime = _StubRuntime(
            {("pc1", "pgrep -a named 2>/dev/null || echo NONE"): "123 named\n"}
        )
        self.assertTrue(runtime.process_running("pc1", "named"))

    def test_process_not_running(self):
        runtime = _StubRuntime(
            {("pc1", "pgrep -a dhcpd 2>/dev/null || echo NONE"): "NONE"}
        )
        self.assertTrue(runtime.process_not_running("pc1", "dhcpd"))

    def test_pidfile_running(self):
        cmd = (
            "if [ -f /tmp/x.pid ] && kill -0 $(cat /tmp/x.pid) 2>/dev/null; "
            "then echo running; else echo not_running; fi"
        )
        runtime = _StubRuntime({("pc1", cmd): "running\n"})
        self.assertTrue(runtime.pidfile_running("pc1", "/tmp/x.pid"))

    def test_interface_exists(self):
        runtime = _StubRuntime({("pc1", "ip link show eth0 2>&1"): "2: eth0: ..."})
        self.assertTrue(runtime.interface_exists("pc1", "eth0"))

    def test_tc_qdisc_contains(self):
        runtime = _StubRuntime(
            {("pc1", "tc qdisc show dev eth0"): "qdisc netem ... corrupt 60%"}
        )
        self.assertTrue(runtime.tc_qdisc_contains("pc1", "eth0", "corrupt"))

    def test_start_background_od_traffic(self):
        addr_json = json.dumps(
            [
                {
                    "ifname": "eth0",
                    "addr_info": [
                        {"family": "inet", "local": "10.0.0.2", "prefixlen": 24}
                    ],
                }
            ]
        )
        runtime = _StubRuntime({("pc2", "ip -j addr"): addr_json})
        labels = runtime.start_background_od_traffic(
            {"pc1": {"pc2": 20}}, interval=10, unit="M"
        )
        self.assertEqual(labels, ["pc1_to_pc2"])
        cmds = [cmd for _, cmd in runtime.calls]
        self.assertTrue(any("iperf3 -s" in cmd for cmd in cmds))
        self.assertTrue(any("iperf3 -c 10.0.0.2" in cmd for cmd in cmds))


if __name__ == "__main__":
    unittest.main()
