"""Bridge NIKA MCP server config to AutoGen StdioServerParams."""

from __future__ import annotations

from autogen_ext.tools.mcp import StdioServerParams

from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers


def to_stdio_params(server: dict) -> StdioServerParams:
    return StdioServerParams(
        command=server["command"],
        args=server.get("args", []),
        env=server.get("env"),
    )


def diagnosis_server_configs(session_id: str, scenario_name: str, problem_names: list[str]) -> dict:
    mcp_cfg = MCPServerConfig(session_id=session_id)
    server_names = select_diagnosis_servers(scenario_name, problem_names)
    return mcp_cfg.load_filtered_config(server_names)


def submission_server_configs(session_id: str) -> dict:
    return MCPServerConfig(session_id=session_id).load_config(if_submit=True)


def diagnosis_server_names(scenario_name: str, problem_names: list[str]) -> list[str]:
    return select_diagnosis_servers(scenario_name, problem_names)


SUBMISSION_SERVER_NAMES = ["task_mcp_server"]
