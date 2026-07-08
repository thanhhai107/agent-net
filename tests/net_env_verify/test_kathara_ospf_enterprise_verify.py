"""Tests for Kathara ospf_enterprise static and dhcp scenarios.

Unit tests (no Docker)
----------------------
- Lab topology structure (routers, hosts, servers).
- ``verify_lab`` signal logic via mocks.

Integration tests (Docker required)
-----------------------------------
- Deploy via ``start_net_env`` (connectivity checked by ``verify_lab``).
- Assert session backend and key nodes are present.

Run via: uv run python -m unittest tests.net_env_verify.test_kathara_ospf_enterprise_verify -v
"""

from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import MagicMock

from nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_dhcp import (
    OSPFEnterpriseDHCP,
)
from nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_static import (
    OSPFEnterpriseStatic,
)
from nika.net_env.kathara.intradomain_routing.ospf_enterprise.verify import (
    DIST_ROUTER_STATIC,
    HOST_GATEWAY,
    HOST_PEER_STATIC_IP,
    HOST_STATIC_IP,
    PEER_HOST,
    PROBE_HOST,
    WEB0_IP,
    WEB0_URL,
    WEB3_URL,
    WEB99_URL,
    verify_ospf_enterprise_lab,
)
from nika.runtime.factory import resolve_backend
from tests.integration_base import SharedSessionTestCase
from tests.net_env_verify.helpers import (
    docker_available,
    instantiate_with_mocked_kathara,
)


class _OSPFEnterpriseVerifyBase(SharedSessionTestCase):
    """Deploy once via ``start_net_env`` (which runs ``verify_lab``); assert topology."""

    __test__ = False

    ENV_RUN_ARGS: ClassVar[list[str]] = ["-s", "s"]

    CORE_ROUTERS = ("router_core_1", "router_core_2", "router_core_3")
    ACCESS_SWITCHES = ("switch_access_1_1_1", "switch_access_2_1_1")
    HOSTS = ("pc_1_1_1_1", "pc_2_1_1_1")

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _OSPFEnterpriseVerifyBase:
            raise unittest.SkipTest("base test class")
        super().setUpClass()

    def _runtime(self):
        from nika.runtime.factory import runtime_for_session

        return runtime_for_session(self._session_row(self.session_id))

    def test_session_uses_kathara_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "kathara")
        self.assertIn(self.SCENARIO, row["lab_name"])

    def _assert_common_key_nodes_deployed(self) -> None:
        nodes = set(self._runtime().list_nodes())
        for name in (
            *self.CORE_ROUTERS,
            *self.ACCESS_SWITCHES,
            *self.HOSTS,
            "dns_server",
        ):
            self.assertIn(name, nodes, f"Expected node {name!r} in deployed lab")

    def test_common_key_nodes_deployed(self) -> None:
        self._assert_common_key_nodes_deployed()


class OSPFEnterpriseLabVerifyUnitTest(unittest.TestCase):
    def _runtime(self, responses: dict[tuple[str, str], str]) -> MagicMock:
        runtime = MagicMock()

        def exec_side_effect(node: str, cmd: str, timeout: float = 10.0) -> str:
            return responses.get((node, cmd), "")

        runtime.exec.side_effect = exec_side_effect
        return runtime

    def _static_responses(self) -> dict[tuple[str, str], str]:
        return {
            ("router_core_1", "vtysh -c 'show ip ospf neighbor'"): "Full\nFull\n",
            ("router_core_1", "vtysh -c 'show ip ospf'"): "Routing Process\n",
            ("router_core_1", "cat /sys/class/net/eth0/operstate"): "up",
            (
                DIST_ROUTER_STATIC,
                "ip -4 -o addr show dev br0",
            ): f"inet {HOST_GATEWAY}/24",
            (PROBE_HOST, "ip -4 -o addr show dev eth0"): f"inet {HOST_STATIC_IP}/24",
            (
                PEER_HOST,
                "ip -4 -o addr show dev eth0",
            ): f"inet {HOST_PEER_STATIC_IP}/24",
            (
                PROBE_HOST,
                "ip route show default",
            ): f"default via {HOST_GATEWAY} dev eth0",
            (PROBE_HOST, f"ping -c 1 -W 2 {HOST_GATEWAY}"): "1 received",
            (PROBE_HOST, f"ping -c 1 -W 2 {HOST_PEER_STATIC_IP}"): "1 received",
            (PROBE_HOST, "ping -c 1 -W 2 10.200.0.2"): "1 received",
            ("dns_server", "systemctl is-active named"): "active",
            (PROBE_HOST, "getent hosts web0.local"): f"{WEB0_IP} web0.local",
            (
                PROBE_HOST,
                f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 {WEB0_URL}",
            ): "200",
            (
                PEER_HOST,
                f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 {WEB3_URL}",
            ): "200",
            ("web_server_0", "systemctl is-active apache2"): "active",
        }

    def _dhcp_responses(self) -> dict[tuple[str, str], str]:
        responses = self._static_responses()
        responses.update(
            {
                (
                    "router_dist_1_1",
                    "ip -4 -o addr show dev br0",
                ): f"inet {HOST_GATEWAY}/24",
                ("router_dist_1_1", "pgrep -a dhcrelay"): "123 dhcrelay -i br0",
                (PROBE_HOST, "ip -4 -o addr show dev eth0"): "inet 10.1.1.42/24",
                (PEER_HOST, "ip -4 -o addr show dev eth0"): "inet 10.2.1.42/24",
                (PROBE_HOST, "ping -c 1 -W 2 10.2.1.42"): "1 received",
                ("dhcp_server", "systemctl is-active isc-dhcp-server"): "active",
                ("web_server_0", "systemctl is-active web_server"): "active",
                ("load_balancer", "pgrep -x nginx"): "456",
                (PROBE_HOST, "getent hosts web99.local"): "10.200.0.10 web99.local",
                (
                    PROBE_HOST,
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 {WEB99_URL}",
                ): "200",
            }
        )
        responses.pop(("web_server_0", "systemctl is-active apache2"), None)
        responses.pop((PROBE_HOST, f"ping -c 1 -W 2 {HOST_PEER_STATIC_IP}"), None)
        return responses

    def test_static_verify_passes_when_all_checks_ok(self) -> None:
        runtime = self._runtime(self._static_responses())
        result = verify_ospf_enterprise_lab(
            runtime, scenario_name="ospf_enterprise_static", mode="static"
        )
        self.assertTrue(result["verified"])
        self.assertTrue(all(result["checks"].values()))

    def test_dhcp_verify_passes_when_all_checks_ok(self) -> None:
        runtime = self._runtime(self._dhcp_responses())
        result = verify_ospf_enterprise_lab(
            runtime, scenario_name="ospf_enterprise_dhcp", mode="dhcp"
        )
        self.assertTrue(result["verified"])
        self.assertTrue(all(result["checks"].values()))

    def test_dhcp_verify_fails_without_dhcp_address(self) -> None:
        responses = self._dhcp_responses()
        responses[(PROBE_HOST, "ip -4 -o addr show dev eth0")] = (
            f"inet {HOST_STATIC_IP}/24"
        )
        result = verify_ospf_enterprise_lab(
            self._runtime(responses),
            scenario_name="ospf_enterprise_dhcp",
            mode="dhcp",
        )
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["host_dhcp_ip"])

    def test_static_verify_fails_when_web_unreachable(self) -> None:
        responses = self._static_responses()
        responses[
            (
                PROBE_HOST,
                f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 {WEB0_URL}",
            )
        ] = "000"
        result = verify_ospf_enterprise_lab(
            self._runtime(responses),
            scenario_name="ospf_enterprise_static",
            mode="static",
        )
        self.assertFalse(result["verified"])
        self.assertFalse(result["checks"]["web_http_web0"])


class OSPFEnterpriseStaticUnitTest(unittest.TestCase):
    def _inst(self) -> OSPFEnterpriseStatic:
        return instantiate_with_mocked_kathara(
            "nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_static.Kathara.get_instance",
            lambda: OSPFEnterpriseStatic(topo_size="s"),
        )

    def test_size_s_key_routers(self) -> None:
        inst = self._inst()
        self.assertEqual(
            set(inst.routers),
            {
                "router_core_1",
                "router_core_2",
                "router_core_3",
                "switch_dist_1_1",
                "switch_dist_2_1",
                "switch_server_access",
            },
        )

    def test_size_s_hosts_and_servers(self) -> None:
        inst = self._inst()
        self.assertEqual(set(inst.hosts), {"pc_1_1_1_1", "pc_2_1_1_1"})
        self.assertIn("dns_server", inst.servers["dns"])
        self.assertEqual(len(inst.servers["web"]), 4)
        self.assertEqual(inst.web_urls, [f"http://web{idx}.local" for idx in range(4)])


class OSPFEnterpriseDHCPUnitTest(unittest.TestCase):
    def _inst(self) -> OSPFEnterpriseDHCP:
        return instantiate_with_mocked_kathara(
            "nika.net_env.kathara.intradomain_routing.ospf_enterprise.lab_dhcp.Kathara.get_instance",
            lambda: OSPFEnterpriseDHCP(topo_size="s"),
        )

    def test_size_s_key_routers(self) -> None:
        inst = self._inst()
        self.assertEqual(
            set(inst.routers),
            {
                "router_core_1",
                "router_core_2",
                "router_core_3",
                "router_dist_1_1",
                "router_dist_2_1",
                "server_access_router",
            },
        )

    def test_size_s_dhcp_and_load_balancer(self) -> None:
        inst = self._inst()
        self.assertIn("dhcp_server", inst.servers["dhcp"])
        self.assertIn("load_balancer", inst.servers["load_balancer"])
        self.assertEqual(len(inst.servers["web"]), 4)
        self.assertIn("http://web99.local", inst.web_urls)


@unittest.skipUnless(docker_available(), "Docker not available")
class OSPFEnterpriseStaticVerifyTest(_OSPFEnterpriseVerifyBase):
    SCENARIO = OSPFEnterpriseStatic.LAB_NAME

    def test_static_key_nodes_deployed(self) -> None:
        self._assert_common_key_nodes_deployed()
        nodes = set(self._runtime().list_nodes())
        for name in (
            "switch_dist_1_1",
            "switch_dist_2_1",
            *(f"web_server_{idx}" for idx in range(4)),
            "switch_server_access",
        ):
            self.assertIn(name, nodes, f"Expected node {name!r} in deployed lab")


@unittest.skipUnless(docker_available(), "Docker not available")
class OSPFEnterpriseDHCPVerifyTest(_OSPFEnterpriseVerifyBase):
    SCENARIO = OSPFEnterpriseDHCP.LAB_NAME

    def test_dhcp_key_nodes_deployed(self) -> None:
        self._assert_common_key_nodes_deployed()
        nodes = set(self._runtime().list_nodes())
        for name in (
            "router_dist_1_1",
            "router_dist_2_1",
            *(f"web_server_{idx}" for idx in range(4)),
            "server_access_router",
            "dhcp_server",
            "load_balancer",
            "backend_web_0",
            "backend_web_1",
            "backend_web_2",
        ):
            self.assertIn(name, nodes, f"Expected node {name!r} in deployed lab")


if __name__ == "__main__":
    unittest.main()
