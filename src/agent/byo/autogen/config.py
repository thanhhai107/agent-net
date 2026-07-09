"""Bridge NIKA MCP server config to AutoGen MCP server params."""

from __future__ import annotations

from autogen_ext.tools.mcp import (
    StdioServerParams,
    StreamableHttpServerParams,
)

from agent.utils.mcp_client import load_session_mcp_config


def to_mcp_params(server: dict) -> StdioServerParams | StreamableHttpServerParams:
    transport = server.get("transport", "stdio")
    if transport == "http":
        return StreamableHttpServerParams(
            url=server["url"],
            headers=dict(server.get("headers") or {}),
        )
    return StdioServerParams(
        command=server["command"],
        args=server.get("args", []),
        env=server.get("env"),
    )


def session_server_configs(session_id: str, scenario_name: str) -> dict:
    return load_session_mcp_config(session_id, scenario_name)


def diagnosis_server_names(scenario_name: str) -> list[str]:
    from agent.utils.mcp_servers import select_diagnosis_servers

    return select_diagnosis_servers(scenario_name)
