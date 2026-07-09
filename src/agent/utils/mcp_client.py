"""Shared MCP client helpers for troubleshooting agents."""

from __future__ import annotations

from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import SUBMISSION
from nika.service.mcp_gateway.phase import advance_mcp_phase


def load_session_mcp_config(
    session_id: str,
    scenario_name: str,
    *,
    backend: str | None = None,
) -> dict:
    """Return session-scoped HTTP MCP config (phase filtering is gateway-side)."""
    return MCPServerConfig(session_id=session_id).load_session_http_config(
        scenario_name,
        backend=backend,
    )


def begin_submission_mcp_phase(session_id: str) -> None:
    """Advance gateway phase before starting the submission workflow step."""
    advance_mcp_phase(session_id, SUBMISSION)
