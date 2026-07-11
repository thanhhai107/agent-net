"""MCP servers exposed to troubleshooting agents."""

from nika.service.mcp_server.registry import (
    MCP_SERVER_PREFIXES,
    MCP_SERVER_SPECS,
    select_diagnosis_servers,
)

__all__ = [
    "MCP_SERVER_PREFIXES",
    "MCP_SERVER_SPECS",
    "select_diagnosis_servers",
]
