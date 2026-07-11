"""Shared MCP client helpers for troubleshooting agents."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from agent.sandbox.config import ENV_SANDBOX_EXECUTION
from agent.utils.mcp_servers import MCPServerConfig, SESSION_HEADER
from agent.utils.phases import SUBMISSION
from nika.service.mcp_gateway.lifecycle import ENV_GATEWAY_AGENT_URL, ENV_GATEWAY_URL
from nika.service.mcp_gateway.phase import advance_mcp_phase


def load_session_mcp_config(
    session_id: str,
    scenario_name: str,
    *,
    backend: str | None = None,
) -> dict:
    """Return session-scoped HTTP MCP config (phase filtering is gateway-side)."""
    if backend is None and os.environ.get("NIKA_SANDBOX_EXECUTION") == "1":
        backend = os.environ.get("NIKA_SESSION_BACKEND", "").strip() or None
    return MCPServerConfig(session_id=session_id).load_session_http_config(
        scenario_name,
        backend=backend,
    )


def _gateway_base_for_phase_advance() -> str:
    if os.environ.get(ENV_SANDBOX_EXECUTION) == "1":
        agent_url = os.environ.get(ENV_GATEWAY_AGENT_URL, "").strip().rstrip("/")
        if agent_url:
            return agent_url
    return os.environ.get(ENV_GATEWAY_URL, "").strip().rstrip("/")


def begin_submission_mcp_phase(session_id: str) -> None:
    """Advance gateway phase before starting the submission workflow step."""
    if os.environ.get(ENV_SANDBOX_EXECUTION) == "1":
        base = _gateway_base_for_phase_advance()
        if not base:
            raise RuntimeError(
                f"{ENV_GATEWAY_AGENT_URL} is not set for sandbox MCP phase advance."
            )
        url = f"{base}/gateway/sessions/{session_id}/phase"
        payload = json.dumps({"phase": SUBMISSION}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                SESSION_HEADER: session_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"MCP phase advance failed with HTTP {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"MCP phase advance failed: HTTP {exc.code}: {body}"
            ) from exc
        return
    advance_mcp_phase(session_id, SUBMISSION)
