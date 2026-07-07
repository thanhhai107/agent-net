import sys

from nika.service.mcp_server.registry import (
    MCP_SERVER_SPECS,
    SUBMISSION_SERVER,
    select_diagnosis_servers,
)

__all__ = ["MCPServerConfig", "select_diagnosis_servers"]


def _resolve_python() -> str:
    """Interpreter for stdio MCP subprocesses (must have ``mcp`` and ``nika`` installed)."""
    return sys.executable or "python3"


class MCPServerConfig:
    def __init__(self, session_id: str):
        if not session_id:
            raise ValueError("session_id is required to start MCP servers.")
        self.session_id = session_id

    def _server_env(self) -> dict[str, str]:
        return {
            "NIKA_SESSION_ID": self.session_id,
        }

    def _build_entry(self, name: str) -> dict:
        spec = MCP_SERVER_SPECS[name]
        return {
            "command": _resolve_python(),
            "args": [spec.script_path],
            "transport": "stdio",
            "env": self._server_env(),
        }

    def load_config(self, if_submit: bool = False) -> dict:
        if if_submit:
            names = [SUBMISSION_SERVER]
        else:
            names = [n for n, spec in MCP_SERVER_SPECS.items() if spec.role != "task"]
        return {name: self._build_entry(name) for name in names}

    def load_filtered_config(self, server_names: list[str]) -> dict:
        """Diagnosis config restricted to *server_names*.

        Unknown names in *server_names* are silently ignored.
        """
        return {
            name: self._build_entry(name)
            for name in server_names
            if name in MCP_SERVER_SPECS
        }
