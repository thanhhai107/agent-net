"""Container entrypoint: load manifest and run the configured agent."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from agent.registry import create_agent
from agent.sandbox.config import ENV_SANDBOX_EXECUTION, ENV_SESSION_DIR
from agent.sandbox.manager import MANIFEST_FILENAME
from nika.service.mcp_gateway.lifecycle import ENV_GATEWAY_AGENT_URL
from nika.utils.agent_config import resolve_max_steps


def main() -> None:
    session_dir = os.environ.get(ENV_SESSION_DIR, "").strip()
    if not session_dir:
        raise SystemExit(f"Missing {ENV_SESSION_DIR} in sandbox container")

    manifest_path = Path(session_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise SystemExit(f"Missing sandbox manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    os.environ[ENV_SANDBOX_EXECUTION] = "1"
    os.environ[ENV_SESSION_DIR] = session_dir
    os.environ.setdefault(
        ENV_GATEWAY_AGENT_URL,
        manifest.get("mcp_gateway_agent_url", ""),
    )
    os.environ.setdefault("NIKA_SESSION_ID", manifest["session_id"])
    backend = str(manifest.get("backend", "")).strip()
    if backend:
        os.environ.setdefault("NIKA_SESSION_BACKEND", backend)

    agent_type = manifest["agent_type"]
    model = manifest["model"]
    max_steps = manifest.get("max_steps")
    if max_steps is None:
        max_steps = resolve_max_steps(None)
    reasoning_effort = manifest.get("reasoning_effort")
    llm_provider = manifest.get("llm_provider")
    stream_output = bool(manifest.get("stream_output", True))
    task_description = manifest["task_description"]

    agent = create_agent(
        agent_type,
        session_id=manifest["session_id"],
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        reasoning_effort=reasoning_effort,
        stream_output=stream_output,
    )
    asyncio.run(agent.run(task_description=task_description))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Sandbox runner failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
