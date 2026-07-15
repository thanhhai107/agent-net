"""Build HTTP MCP client configs for NIKA troubleshooting agents."""

from __future__ import annotations

import os

from nika.service.mcp_gateway.lifecycle import ENV_GATEWAY_URL
from nika.service.mcp_server.registry import (
    MCP_SERVER_SPECS,
    SUBMISSION_SERVER,
    select_diagnosis_servers,
)

__all__ = [
    "MCPServerConfig",
    "select_diagnosis_servers",
    "select_session_servers",
    "session_http_headers",
]

SESSION_HEADER = "NIKA-Session-Id"


def session_http_headers(session_id: str) -> dict[str, str]:
    return {SESSION_HEADER: session_id}


def _gateway_base_url() -> str:
    base = os.environ.get(ENV_GATEWAY_URL, "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            f"{ENV_GATEWAY_URL} is not set. Start the MCP gateway before building HTTP config."
        )
    return base


def select_session_servers(
    scenario_name: str,
    *,
    backend: str | None = None,
) -> list[str]:
    """Return all MCP server names for a troubleshooting session."""
    servers = select_diagnosis_servers(
        scenario_name,
        backend=backend,
    )
    if SUBMISSION_SERVER not in servers:
        servers.append(SUBMISSION_SERVER)
    return servers


class MCPServerConfig:
    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id is required to start MCP servers.")
        self.session_id = session_id

    def _build_http_entry(self, name: str) -> dict:
        if name not in MCP_SERVER_SPECS:
            raise KeyError(f"Unknown MCP server: {name!r}")
        base = _gateway_base_url()
        return {
            "transport": "http",
            "url": f"{base}/mcp/{name}/mcp",
            "headers": session_http_headers(self.session_id),
        }

    def load_http_config(self, server_names: list[str]) -> dict:
        """Return HTTP MCP client config for *server_names*."""
        return {
            name: self._build_http_entry(name)
            for name in server_names
            if name in MCP_SERVER_SPECS
        }

    def load_session_http_config(
        self,
        scenario_name: str,
        *,
        backend: str | None = None,
    ) -> dict:
        """Return HTTP MCP config for all servers needed by the session."""
        server_names = select_session_servers(
            scenario_name,
            backend=backend,
        )
        return self.load_http_config(server_names)
