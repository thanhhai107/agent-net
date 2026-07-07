"""Kathara API smoke tests on ``ospf_enterprise_static`` (small topology).

Exercises OSPF FRR parsers, DNS lookup, and HTTP curl helpers that need a
multi-tier enterprise lab rather than ``simple_bgp``.

Run:
  uv run python -m unittest tests.api_verify.test_kathara_ospf_api -v
"""

from __future__ import annotations

import unittest
from typing import ClassVar

from nika.net_env.kathara.intradomain_routing.ospf_enterprise.verify import (
    CORE_ROUTER,
    DNS_SERVER,
    PROBE_HOST,
    WEB0_URL,
    WEB3_URL,
)
from nika.runtime.factory import resolve_backend
from tests.api_verify.helpers import docker_available
from tests.api_verify.kathara_base import KatharaScenarioApiSmokeTest

HOST = PROBE_HOST
ROUTER = CORE_ROUTER
INTF = "eth0"


@unittest.skipUnless(docker_available(), "Docker not available")
class KatharaOspfApiSmokeTest(KatharaScenarioApiSmokeTest):
    SCENARIO = "ospf_enterprise_static"
    ENV_RUN_ARGS: ClassVar[list[str]] = ["-s", "s"]

    def test_session_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "kathara")

    def test_runtime_ospf_semantic_apis(self) -> None:
        runtime = self._runtime()
        self.assertTrue(
            self.smoke(
                "runtime.process_running(ospfd)",
                lambda: runtime.process_running(ROUTER, "ospfd"),
                expect_type=bool,
            )
        )
        self.smoke(
            "runtime.dig_query(web0.local)",
            lambda: runtime.dig_query(HOST, "web0.local"),
            min_len=1,
        )

    def test_kathara_frr_ospf_api(self) -> None:
        api = self._frr_api()
        self.smoke(
            "KatharaFRRAPI.frr_get_ospf_conf",
            lambda: api.frr_get_ospf_conf(ROUTER),
            min_len=1,
        )
        neighbors = self.smoke(
            "KatharaFRRAPI.frr_get_ospf_neighbors",
            lambda: api.frr_get_ospf_neighbors(ROUTER),
            min_len=1,
        )
        self.assertIn("Full", neighbors)
        self.smoke(
            "KatharaFRRAPI.frr_get_ospf_routes",
            lambda: api.frr_get_ospf_routes(ROUTER),
            min_len=1,
        )
        self.smoke(
            "KatharaFRRAPI.frr_get_ospf_interfaces",
            lambda: api.frr_get_ospf_interfaces(ROUTER),
            min_len=1,
        )
        self.smoke(
            "KatharaFRRAPI.frr_exec(show ip ospf)",
            lambda: api.frr_exec(ROUTER, "show ip ospf"),
            min_len=1,
        )
        self.smoke(
            "KatharaFRRAPI.frr_show_route",
            lambda: api.frr_show_route(ROUTER),
            min_len=1,
        )

    def test_kathara_host_dns_and_web_api(self) -> None:
        api = self._host_api()
        dns_cfg = self.smoke(
            "KatharaBaseAPI.show_dns_config",
            lambda: api.show_dns_config(HOST),
            min_len=1,
        )
        self.assertIn("nameserver", dns_cfg.lower())
        self.smoke(
            "KatharaBaseAPI.curl_web_test(web0)",
            lambda: api.curl_web_test(HOST, WEB0_URL, times=1),
            min_len=1,
        )
        self.smoke(
            "KatharaBaseAPI.curl_web_test(web3)",
            lambda: api.curl_web_test(HOST, WEB3_URL, times=1),
            min_len=1,
        )
        self.smoke(
            "KatharaBaseAPI.systemctl_ops(named)",
            lambda: api.systemctl_ops(DNS_SERVER, "named", "status"),
            min_len=1,
        )
        hosts = self.smoke("KatharaBaseAPI.get_hosts", api.get_hosts, expect_type=list)
        self.assertIn(HOST, hosts)


if __name__ == "__main__":
    unittest.main()
