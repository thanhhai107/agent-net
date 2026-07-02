"""Bridge NIKA MCP server config to mcp-agent Settings."""

from __future__ import annotations

from mcp_agent.config import MCPServerSettings, MCPSettings, OpenAISettings, Settings

from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers


def _to_server_settings(server: dict) -> MCPServerSettings:
    return MCPServerSettings(
        transport=server.get("transport", "stdio"),
        command=server["command"],
        args=server.get("args", []),
        env=server.get("env"),
    )


def build_mcp_agent_settings(
    session_id: str,
    scenario_name: str,
    problem_names: list[str],
    model: str,
) -> Settings:
    """Build mcp-agent Settings for a NIKA troubleshooting session."""
    mcp_cfg = MCPServerConfig(session_id=session_id)
    diag_cfg = mcp_cfg.load_filtered_config(
        select_diagnosis_servers(scenario_name, problem_names)
    )
    submit_cfg = mcp_cfg.load_config(if_submit=True)
    servers = {**diag_cfg, **submit_cfg}

    return Settings(
        execution_engine="asyncio",
        mcp=MCPSettings(
            servers={name: _to_server_settings(srv) for name, srv in servers.items()}
        ),
        openai=OpenAISettings(default_model=model),
    )


def diagnosis_server_names(scenario_name: str, problem_names: list[str]) -> list[str]:
    return select_diagnosis_servers(scenario_name, problem_names)


SUBMISSION_SERVER_NAMES = ["task_mcp_server"]
