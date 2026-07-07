"""Unit tests for containerlab SRL MCP server tools."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from nika.service.mcp_server.containerlab import srl_server as srl_mcp


class ContainerlabSrlMcpServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = MagicMock()
        self.api.srl_exec_cli.return_value = "route output"
        self.api.srl_get_bgp_as.return_value = 65001

    @patch.object(srl_mcp, "get_srl_api")
    def test_srl_exec_cli(self, mock_get_api: MagicMock) -> None:
        mock_get_api.return_value = self.api
        result = srl_mcp.srl_exec_cli("leaf1", "show version")
        self.api.srl_exec_cli.assert_called_once_with("leaf1", "show version")
        self.assertEqual(result, "route output")

    @patch.object(srl_mcp, "get_srl_api")
    def test_srl_get_bgp_as(self, mock_get_api: MagicMock) -> None:
        mock_get_api.return_value = self.api
        result = srl_mcp.srl_get_bgp_as("leaf1")
        self.api.srl_get_bgp_as.assert_called_once_with("leaf1")
        self.assertEqual(result, 65001)

    @patch.object(srl_mcp, "get_srl_api")
    def test_srl_show_running_config(self, mock_get_api: MagicMock) -> None:
        mock_get_api.return_value = self.api
        srl_mcp.srl_show_running_config("leaf1")
        self.api.srl_exec_cli.assert_called_once_with("leaf1", "info from running")

    @patch.object(srl_mcp, "get_srl_api")
    def test_srl_show_bgp_summary(self, mock_get_api: MagicMock) -> None:
        mock_get_api.return_value = self.api
        srl_mcp.srl_show_bgp_summary("leaf1")
        self.api.srl_exec_cli.assert_called_once_with(
            "leaf1",
            "show network-instance default protocols bgp summary",
        )

    @patch.object(srl_mcp, "get_srl_api")
    def test_srl_show_ip_route(self, mock_get_api: MagicMock) -> None:
        mock_get_api.return_value = self.api
        srl_mcp.srl_show_ip_route("leaf1")
        self.api.srl_exec_cli.assert_called_once_with(
            "leaf1",
            "show network-instance default route-table ipv4",
        )


if __name__ == "__main__":
    unittest.main()
