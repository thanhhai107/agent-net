"""Containerlab min3clos end-to-end pipeline integration test."""

from __future__ import annotations

import asyncio
import unittest
from typing import ClassVar

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.utils.mcp_servers import MCPServerConfig
from nika.service.mcp_gateway.lifecycle import mcp_gateway_for_session
from tests.nika.workflows.integration import pipeline_case
from tests.support.prerequisites import containerlab_prerequisites

MIN3CLOS_NODES = frozenset({"leaf1", "leaf2", "spine", "client1", "client2"})


@unittest.skipUnless(
    containerlab_prerequisites(), "containerlab, gnmic, or Docker not available"
)
class ClabPipelineIntegrationTest(pipeline_case.PipelineCaseBase):
    SCENARIO = "min3clos"
    BACKEND = "containerlab"
    ENV_RUN_ARGS: ClassVar[list[str]] = []
    PROBLEM = "link_down"
    INJECT_PARAMS = {"host_name": "leaf1", "intf_name": "e1-1"}
    EXPECTED_NODES = MIN3CLOS_NODES
    EXEC_PROBE_HOST = "client1"
    SUBMIT_FAULTY_DEVICES = ["leaf1"]
    IMAGE_SUBSTRING = None
    DIAGNOSIS_MCP_SERVERS = [
        "kathara_base_mcp_server",
        "containerlab_srl_mcp_server",
    ]

    def test_step_05_diagnosis_mcp_tools(self) -> None:
        self.assertIsNotNone(self.session_id)
        with mcp_gateway_for_session(
            self.session_id,
            scenario_name=self.SCENARIO,
        ):
            diagnosis_config = MCPServerConfig(
                session_id=self.session_id
            ).load_http_config(self.DIAGNOSIS_MCP_SERVERS)

            async def _run() -> dict:
                client = MultiServerMCPClient(connections=diagnosis_config)
                tools = {t.name: t for t in await client.get_tools()}
                reach = await tools["get_reachability"].ainvoke({})
                host_cfg = await tools["get_host_net_config"].ainvoke(
                    {"host_name": self.EXEC_PROBE_HOST}
                )
                exec_out = await tools["exec_shell"].ainvoke(
                    {"host_name": self.EXEC_PROBE_HOST, "command": self.EXEC_PROBE_CMD}
                )
                bgp_as = await tools["srl_get_bgp_as"].ainvoke({"device_name": "leaf1"})
                routes = await tools["srl_show_ip_route"].ainvoke(
                    {"device_name": "leaf1"}
                )
                return {
                    "reachability": str(reach),
                    "host_net_config": str(host_cfg),
                    "exec_shell": str(exec_out),
                    "srl_get_bgp_as": str(bgp_as),
                    "srl_show_ip_route": str(routes),
                }

            results = asyncio.run(_run())
        for key, output in results.items():
            self.assertTrue(len(output) > 0, f"{key} must return non-empty output")
            self.assertNotIn("NIKA_SESSION_ID is not set", output)


if __name__ == "__main__":
    unittest.main()
