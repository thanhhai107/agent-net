"""Unit tests for MCP server selection."""

from __future__ import annotations

import unittest

from agent.utils.mcp_servers import select_diagnosis_servers


class SelectDiagnosisServersTest(unittest.TestCase):
    def test_min3clos_skips_frr_mcp(self) -> None:
        servers = select_diagnosis_servers("min3clos", ["bgp_asn_misconfig"])
        self.assertIn("kathara_base_mcp_server", servers)
        self.assertNotIn("kathara_frr_mcp_server", servers)

    def test_min3clos_bgp_includes_srl_mcp(self) -> None:
        servers = select_diagnosis_servers("min3clos", ["bgp_asn_misconfig"])
        self.assertIn("containerlab_srl_mcp_server", servers)

    def test_dc_clos_bgp_includes_frr_mcp(self) -> None:
        servers = select_diagnosis_servers("dc_clos_bgp", ["bgp_asn_misconfig"])
        self.assertIn("kathara_frr_mcp_server", servers)

    def test_explicit_containerlab_backend(self) -> None:
        servers = select_diagnosis_servers(
            "simple_bgp", ["bgp_asn_misconfig"], backend="containerlab"
        )
        self.assertNotIn("kathara_frr_mcp_server", servers)


if __name__ == "__main__":
    unittest.main()
