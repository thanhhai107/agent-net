"""Helpers for sandbox security and integration tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from agent.sandbox.config import (
    DEFAULT_SANDBOX_IMAGE,
    SANDBOX_NETWORK_HOST,
    resolve_sandbox_config,
    sandbox_gateway_agent_host,
)
from agent.sandbox.env import build_sandbox_env, format_env_for_log
from agent.sandbox.image import ensure_sandbox_image
from nika.service.mcp_gateway.lifecycle import (
    ENV_GATEWAY_AGENT_URL,
    mcp_gateway_for_session,
)

SECURITY_PROBE_SCRIPT = Path(__file__).resolve().parent / "security_probe.sh"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def sandbox_image_available(image: str = DEFAULT_SANDBOX_IMAGE) -> bool:
    if not docker_available():
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def run_security_probe(
    *,
    image: str = DEFAULT_SANDBOX_IMAGE,
    network: str = SANDBOX_NETWORK_HOST,
    gateway_agent_url: str | None = None,
    host_probe_file: Path | None = None,
    env_dump: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the security probe script inside a sandbox-style container."""
    if not SECURITY_PROBE_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing probe script: {SECURITY_PROBE_SCRIPT}")

    workspace = tempfile.mkdtemp(prefix="nika-sandbox-probe-")
    cmd = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--user",
        "agent",
        "--network",
        network,
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:/workspace:rw",
        "-v",
        f"{SECURITY_PROBE_SCRIPT}:/tmp/security_probe.sh:ro",
    ]

    if network != SANDBOX_NETWORK_HOST:
        cmd.extend(["--add-host=host.docker.internal:host-gateway"])

    if host_probe_file is not None:
        cmd.extend(["-e", f"NIKA_SANDBOX_PROBE_FILE={host_probe_file}"])
    if gateway_agent_url:
        cmd.extend(["-e", f"NIKA_MCP_GATEWAY_AGENT_URL={gateway_agent_url}"])

    if env_dump:
        dump_path = Path(workspace) / "env_dump.txt"
        dump_path.write_text(
            "\n".join(f"{k}={v}" for k, v in format_env_for_log(env_dump).items()),
            encoding="utf-8",
        )

    cmd.extend([image, "bash", "/tmp/security_probe.sh"])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def run_security_probe_with_gateway(session_id: str = "sandbox-security-test") -> None:
    """Start a gateway and verify container probes against it."""
    config = resolve_sandbox_config(enabled=True, network=SANDBOX_NETWORK_HOST)
    ensure_sandbox_image(
        config.image,
        http_proxy=config.http_proxy,
        https_proxy=config.https_proxy,
    )
    agent_host = sandbox_gateway_agent_host(config.network)
    with mcp_gateway_for_session(
        session_id,
        scenario_name="simple_bgp",
        sandbox=True,
        sandbox_agent_host=agent_host,
    ):
        gateway_url = os.environ[ENV_GATEWAY_AGENT_URL]
        host_probe = Path(tempfile.mkdtemp()) / "host_secret_probe.txt"
        host_probe.write_text("host-only", encoding="utf-8")
        env = build_sandbox_env(
            session_id=session_id,
            session_dir="/tmp/unused",
            agent_type="local_cli.codex_cli",
            model="gpt-5.4-mini",
            max_steps=5,
            reasoning_effort=None,
            llm_provider=None,
            mcp_gateway_agent_url=gateway_url,
            env_file=config.env_file,
            skills_dir="/nika/skills",
            http_proxy=config.http_proxy,
            https_proxy=config.https_proxy,
            no_proxy=config.no_proxy,
        )
        result = run_security_probe(
            image=config.image,
            network=config.network,
            gateway_agent_url=gateway_url,
            host_probe_file=host_probe,
            env_dump=env,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Security probe failed (code {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
