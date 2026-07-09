"""Unit tests for PingMesh MCP server tools."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from nika.service.mcp_server.common import pingmesh_server as pingmesh_mcp


class PingmeshMcpServerTest(unittest.IsolatedAsyncioTestCase):
    @patch.object(pingmesh_mcp, "get_lab_api")
    @patch.object(pingmesh_mcp, "execute_pingmesh_snapshot", new_callable=AsyncMock)
    async def test_run_pingmesh_snapshot(
        self,
        mock_execute: AsyncMock,
        mock_get_api: MagicMock,
    ) -> None:
        mock_get_api.return_value = MagicMock()
        mock_execute.return_value = {
            "timestamp": "2026-07-09T00:00:00+00:00",
            "endpoints": {"pc1": "10.0.0.1", "pc2": "10.0.0.2"},
            "sources": ["pc1", "pc2"],
            "targets": ["pc1", "pc2"],
            "results": [],
            "anomalies": [],
            "summary": {
                "total_pairs": 0,
                "reachable_pairs": 0,
                "anomaly_count": 0,
                "unreachable": 0,
                "packet_loss": 0,
                "high_latency": 0,
                "unknown": 0,
            },
        }
        result = await pingmesh_mcp.run_pingmesh_snapshot(count=2)
        payload = json.loads(result)
        self.assertIn("summary", payload)
        mock_execute.assert_awaited_once()
        kwargs = mock_execute.await_args.kwargs
        self.assertEqual(kwargs["count"], 2)


if __name__ == "__main__":
    unittest.main()
