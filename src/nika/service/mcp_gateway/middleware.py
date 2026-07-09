"""ASGI middleware for session binding and phase gating."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from nika.service.mcp_gateway.context import bind_session, reset_session
from nika.service.mcp_gateway.policy import is_server_allowed

SESSION_HEADER = "NIKA-Session-Id"
_MCP_JSON = "application/json"

_empty_mcp = FastMCP("nika_phase_blocked")


class PhaseGateMiddleware:
    """Bind session context and enforce phase policy for one MCP server mount."""

    def __init__(self, app, *, server_name: str, blocked_app):
        self.app = app
        self.blocked_app = blocked_app
        self.server_name = server_name

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        session_id = headers.get(b"nika-session-id", b"").decode().strip()

        if not session_id:
            await _send_json(
                send,
                status=400,
                payload={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": f"Missing {SESSION_HEADER} header.",
                    },
                    "id": None,
                },
            )
            return

        target_app = (
            self.app
            if is_server_allowed(session_id, self.server_name)
            else self.blocked_app
        )

        token = bind_session(session_id)
        try:
            await target_app(scope, receive, send)
        finally:
            reset_session(token)


async def _send_json(send, *, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", _MCP_JSON.encode()),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
