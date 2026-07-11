"""Run troubleshooting agents inside a Docker sandbox container."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent.sandbox.config import SandboxConfig, sandbox_uses_host_network
from agent.sandbox.env import build_sandbox_env, format_env_for_log
from agent.sandbox.redact import redact_text
from agent.utils.skills import resolve_skills_root
from nika.utils.logger import log_event
from nika.utils.session import Session

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "sandbox_manifest.json"
CONTAINER_SKILLS_DIR = "/nika/skills"
CONTAINER_CODEX_AUTH = "/home/agent/.codex/auth.json"


@dataclass
class SandboxRunResult:
    returncode: int
    container_id: str | None = None


class SandboxManager:
    """Launch and monitor the agent sandbox container."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    def write_manifest(
        self,
        *,
        session: Session,
        agent_type: str,
        model: str,
        max_steps: int | None,
        reasoning_effort: str | None,
        llm_provider: str | None,
        mcp_gateway_agent_url: str,
        stream_output: bool,
    ) -> Path:
        manifest = {
            "session_id": session.session_id,
            "session_dir": session.session_dir,
            "agent_type": agent_type,
            "model": model,
            "max_steps": max_steps,
            "reasoning_effort": reasoning_effort,
            "llm_provider": llm_provider,
            "task_description": session.task_description,
            "scenario_name": getattr(session, "scenario_name", ""),
            "backend": getattr(session, "backend", "") or "kathara",
            "mcp_gateway_agent_url": mcp_gateway_agent_url.rstrip("/"),
            "stream_output": stream_output,
        }
        path = Path(session.session_dir) / MANIFEST_FILENAME
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    def build_docker_command(
        self,
        *,
        session: Session,
        agent_type: str,
        model: str,
        max_steps: int | None,
        reasoning_effort: str | None,
        llm_provider: str | None,
        mcp_gateway_agent_url: str,
    ) -> list[str]:
        session_dir = str(Path(session.session_dir).resolve())
        skills_host = str(resolve_skills_root().resolve())

        env = build_sandbox_env(
            session_id=session.session_id,
            session_dir=session_dir,
            agent_type=agent_type,
            model=model,
            max_steps=max_steps,
            reasoning_effort=reasoning_effort,
            llm_provider=llm_provider,
            mcp_gateway_agent_url=mcp_gateway_agent_url,
            env_file=self.config.env_file,
            skills_dir=CONTAINER_SKILLS_DIR,
            http_proxy=self.config.http_proxy,
            https_proxy=self.config.https_proxy,
            no_proxy=self.config.no_proxy,
        )
        backend = getattr(session, "backend", "").strip()
        if backend:
            env["NIKA_SESSION_BACKEND"] = backend

        cmd: list[str] = [
            "docker",
            "run",
            "--init",
            "--user",
            "agent",
            "--network",
            self.config.network,
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{session_dir}:{session_dir}:rw",
            "-v",
            f"{skills_host}:{CONTAINER_SKILLS_DIR}:ro",
        ]

        if not sandbox_uses_host_network(self.config.network):
            cmd.extend(["--add-host=host.docker.internal:host-gateway"])

        if not self.config.keep_container:
            cmd.append("--rm")
        if self.config.cpus:
            cmd.extend(["--cpus", self.config.cpus])
        if self.config.memory:
            cmd.extend(["--memory", self.config.memory])

        if (
            agent_type in ("local_cli.codex_cli", "sdk.codex_sdk")
            and self.config.codex_auth_file
            and self.config.codex_auth_file.is_file()
        ):
            cmd.extend(
                [
                    "-v",
                    f"{self.config.codex_auth_file}:{CONTAINER_CODEX_AUTH}:ro",
                ]
            )

        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend(
            [
                self.config.image,
                "python",
                "-m",
                "agent.sandbox.runner",
            ]
        )
        return cmd

    def run(
        self,
        *,
        session: Session,
        agent_type: str,
        model: str,
        max_steps: int | None,
        reasoning_effort: str | None,
        llm_provider: str | None,
        mcp_gateway_agent_url: str,
        stream_output: bool = True,
    ) -> SandboxRunResult:
        self.write_manifest(
            session=session,
            agent_type=agent_type,
            model=model,
            max_steps=max_steps,
            reasoning_effort=reasoning_effort,
            llm_provider=llm_provider,
            mcp_gateway_agent_url=mcp_gateway_agent_url,
            stream_output=stream_output,
        )

        cmd = self.build_docker_command(
            session=session,
            agent_type=agent_type,
            model=model,
            max_steps=max_steps,
            reasoning_effort=reasoning_effort,
            llm_provider=llm_provider,
            mcp_gateway_agent_url=mcp_gateway_agent_url,
        )

        env_preview = build_sandbox_env(
            session_id=session.session_id,
            session_dir=str(Path(session.session_dir).resolve()),
            agent_type=agent_type,
            model=model,
            max_steps=max_steps,
            reasoning_effort=reasoning_effort,
            llm_provider=llm_provider,
            mcp_gateway_agent_url=mcp_gateway_agent_url,
            env_file=self.config.env_file,
            skills_dir=CONTAINER_SKILLS_DIR,
            http_proxy=self.config.http_proxy,
            https_proxy=self.config.https_proxy,
            no_proxy=self.config.no_proxy,
        )
        log_event(
            "sandbox_start",
            f"Starting sandbox container for session {session.session_id}",
            session_id=session.session_id,
            agent_type=agent_type,
            image=self.config.image,
            network=self.config.network,
            http_proxy=self.config.http_proxy,
            docker_command=redact_text(" ".join(cmd[:12] + ["..."])),
            env=format_env_for_log(env_preview),
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if stream_output:
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"Sandbox agent exited with code {returncode} for session {session.session_id}"
            )
        log_event(
            "sandbox_end",
            f"Sandbox container finished for session {session.session_id}",
            session_id=session.session_id,
            agent_type=agent_type,
            returncode=returncode,
        )
        return SandboxRunResult(returncode=returncode)
