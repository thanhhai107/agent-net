import sys

from nika.config import MCP_SERVER_DIR

# Keyword sets that trigger inclusion of each optional Kathara MCP server.
_FRR_KEYWORDS = frozenset({"bgp", "ospf", "rip", "frr", "routing"})
_BMV2_KEYWORDS = frozenset({"p4", "bmv2", "sdn", "bloom", "mpls", "int", "counter"})
_TELEMETRY_KEYWORDS = frozenset({"telemetry", "influx", "int"})


def select_diagnosis_servers(
    scenario_name: str,
) -> list[str]:
    """Return the minimal public Kathara MCP server set needed for *scenario*.

    ``kathara_base_mcp_server`` is always included.  The three specialised
    servers are added only when keyword signals appear in the public scenario
    name (tokens are split on ``_`` and ``-``). Hidden injected problem labels
    are deliberately ignored so evaluation remains fair.

    Parameters
    ----------
    scenario_name:
        E.g. ``"dc_clos_bgp"`` or ``"p4_counter"``.
    """
    combined = scenario_name.lower()
    tokens = set(combined.replace("_", " ").replace("-", " ").split())

    servers = ["kathara_base_mcp_server"]
    clos_service_routing = "clos" in tokens and "sdn" not in tokens
    if tokens & _FRR_KEYWORDS or clos_service_routing:
        servers.append("kathara_frr_mcp_server")
    if tokens & _BMV2_KEYWORDS:
        servers.append("kathara_bmv2_mcp_server")
    if tokens & _TELEMETRY_KEYWORDS:
        servers.append("kathara_telemetry_mcp_server")
    return servers


class MCPServerConfig:
    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id is required to start MCP servers.")
        self.mcp_server_dir = str(MCP_SERVER_DIR)
        self.session_id = session_id

    def _server_env(self, **extra: str) -> dict[str, str]:
        return {
            "NIKA_SESSION_ID": self.session_id,
            **extra,
        }

    def load_config(self, if_submit: bool = False) -> dict:
        if if_submit:
            config = {
                "task_mcp_server": {
                    "command": sys.executable,
                    "args": [f"{self.mcp_server_dir}/task_mcp_server.py"],
                    "transport": "stdio",
                },
            }
        else:
            config = {
                "kathara_base_mcp_server": {
                    "command": sys.executable,
                    "args": [f"{self.mcp_server_dir}/kathara_base_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_frr_mcp_server": {
                    "command": sys.executable,
                    "args": [f"{self.mcp_server_dir}/kathara_frr_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_bmv2_mcp_server": {
                    "command": sys.executable,
                    "args": [f"{self.mcp_server_dir}/kathara_bmv2_mcp_server.py"],
                    "transport": "stdio",
                },
                "kathara_telemetry_mcp_server": {
                    "command": sys.executable,
                    "args": [f"{self.mcp_server_dir}/kathara_telemetry_mcp_server.py"],
                    "transport": "stdio",
                },
            }

        for server in config.values():
            server["env"] = self._server_env()
        return config

    def load_filtered_config(self, server_names: list[str]) -> dict:
        """Diagnosis config restricted to *server_names*.

        Useful when only a subset of Kathara MCP servers is relevant for a
        given scenario (e.g. skip bmv2 tools for a pure routing problem).
        Unknown names in *server_names* are silently ignored.
        """
        full = self.load_config(if_submit=False)
        return {k: v for k, v in full.items() if k in server_names}

    def load_tool_docs_config(self, library_id: str) -> dict:
        """Return the FastMCP adapter for one DRAFT documentation library."""
        if not library_id:
            raise ValueError("library_id is required for DRAFT tool docs.")
        return {
            "nika_tool_docs": {
                "command": sys.executable,
                "args": [
                    f"{self.mcp_server_dir}/tool_evolution_mcp_server.py"
                ],
                "transport": "stdio",
                "env": self._server_env(NIKA_TOOL_LIBRARY_ID=library_id),
            }
        }
