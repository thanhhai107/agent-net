"""Host-side MCP HTTP gateway for NIKA troubleshooting agents."""

from nika.service.mcp_gateway.lifecycle import (
    McpGatewayManager,
    mcp_gateway_for_session,
    start_gateway,
    stop_gateway,
)
from nika.service.mcp_gateway.phase import advance_mcp_phase

__all__ = [
    "McpGatewayManager",
    "advance_mcp_phase",
    "mcp_gateway_for_session",
    "start_gateway",
    "stop_gateway",
]
