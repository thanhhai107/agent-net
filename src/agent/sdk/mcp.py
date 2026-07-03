"""Shared MCP config adapters for SDK-based agents."""

from __future__ import annotations

import sys
from typing import Any


def _resolve_python() -> str:
    """Interpreter used to spawn stdio MCP servers."""
    return sys.executable or "python3"


def to_sdk_mcp_servers(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt NIKA's MultiServerMCPClient config to claude-agent-sdk stdio format.

    NIKA returns ``{"transport": "stdio", "command": ..., "args": ...}`` (the
    langchain-mcp-adapters shape); claude-agent-sdk expects
    ``{"type": "stdio", "command": ..., "args": ..., "env": ...}``.
    """
    servers: dict[str, Any] = {}
    for name, spec in config.items():
        command = spec.get("command")
        if command in ("python3", "python"):
            command = _resolve_python()
        servers[name] = {
            "type": "stdio",
            "command": command,
            "args": list(spec.get("args", [])),
            "env": dict(spec.get("env", {})),
        }
    return servers
