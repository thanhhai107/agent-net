"""Codex CLI subprocess adapter for LangGraph nodes.

Each ``CodexWorker`` instance drives one ``codex exec`` invocation inside an
isolated, per-session workspace.  It handles:

* **Workspace creation** – ``{session_dir}/codex_workspace/`` (git-initialised
  so Codex is happy; safe to call multiple times).
* **CODEX_HOME isolation** – a private ``.codex_home/`` inside the workspace is
  used as ``CODEX_HOME``, so no files are written to ``~/.codex/``.
  ``auth.json`` is sym-linked from the user's real ``~/.codex/auth.json`` so
  that authentication still works.
* **MCP server config** – ``config.toml`` in the isolated home contains only
  the servers relevant to the current phase and scenario (selected by
  :func:`~agent.utils.mcp_servers.select_diagnosis_servers`).
* **Session ID propagation** – ``NIKA_SESSION_ID`` is injected into every MCP
  server's ``env`` block, exactly as :class:`~agent.utils.mcp_servers.MCPServerConfig`
  does for the LangChain path.
* **Output capture** – the final assistant message is written by
  ``--output-last-message``; JSONL events emitted via ``--json`` are streamed
  line-by-line, logged to ``messages.jsonl`` in real time, and pretty-printed to
  the terminal via :func:`~agent.local_cli.codex_cli.codex_display.format_codex_event`.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from agent.local_cli.codex_cli.codex_display import format_codex_event
from agent.utils.loggers import MessageLogger
from agent.utils.mcp_client import begin_submission_mcp_phase, load_session_mcp_config
from agent.utils.phases import PHASES, SUBMISSION
from agent.utils.skills import prepare_codex_workspace

REASONING_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")

# ---------------------------------------------------------------------------
# TOML helper
# ---------------------------------------------------------------------------


def _build_mcp_toml(servers: dict) -> str:
    """Serialise an MCP server dict (from MCPServerConfig) as TOML."""
    lines: list[str] = [
        "experimental_use_rmcp_client = true",
        'approval_policy = "never"',
        'sandbox_mode = "workspace-write"',
        "",
    ]
    for name, srv in servers.items():
        lines.append(f"[mcp_servers.{name}]")
        if srv.get("transport") == "http":
            lines.append(f'url = "{srv["url"]}"')
            lines.append('default_tools_approval_mode = "approve"')
            headers: dict = srv.get("headers") or {}
            if headers:
                lines.append(f"\n[mcp_servers.{name}.http_headers]")
                for k, v in headers.items():
                    lines.append(f'{k} = "{v}"')
        else:
            lines.append(f'command = "{srv["command"]}"')
            args_toml = "[" + ", ".join(f'"{a}"' for a in srv["args"]) + "]"
            lines.append(f"args = {args_toml}")
            lines.append('default_tools_approval_mode = "approve"')
            env: dict = srv.get("env", {})
            if env:
                lines.append(f"\n[mcp_servers.{name}.env]")
                for k, v in env.items():
                    lines.append(f'{k} = "{v}"')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CodexWorker
# ---------------------------------------------------------------------------


class CodexWorker:
    """Run one non-interactive ``codex exec`` invocation as a LangGraph node.

    Parameters
    ----------
    session_id:
        NIKA session identifier — resolves the session directory and is
        propagated to MCP servers via ``NIKA_SESSION_ID``.
    session_dir:
        Absolute path to the session results directory.
    phase:
        One of :data:`~agent.utils.phases.PHASES` (``diagnosis`` or ``submission``).
    model:
        Codex model name forwarded to ``codex exec -m``.
    reasoning_effort:
        Optional Codex ``model_reasoning_effort`` override forwarded via
        ``codex exec -c model_reasoning_effort=...``.
    timeout:
        Hard timeout in seconds for the subprocess (default 600 s).
    scenario_name:
        Used by :func:`~agent.utils.mcp_servers.select_diagnosis_servers` to pick relevant servers.
        Ignored for the submission phase (which always uses the task server).
    """

    def __init__(
        self,
        session_id: str,
        session_dir: str,
        phase: str,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str | None = None,
        timeout: int = 600,
        scenario_name: str = "",
        *,
        stream_output: bool = True,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        if (
            reasoning_effort is not None
            and reasoning_effort not in REASONING_EFFORT_LEVELS
        ):
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORT_LEVELS}, got {reasoning_effort!r}"
            )

        self.session_id = session_id
        self.phase = phase
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.scenario_name = scenario_name

        self.workspace = Path(session_dir) / "codex_workspace"
        self._codex_home = self.workspace / ".codex_home"
        self._logger = MessageLogger(agent=phase, session_dir=session_dir)
        self._stream_output = stream_output

    # ------------------------------------------------------------------
    # Workspace + isolated CODEX_HOME setup
    # ------------------------------------------------------------------

    def _setup_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._codex_home.mkdir(parents=True, exist_ok=True)

        # Initialise a git repo so Codex doesn't complain.
        if not (self.workspace / ".git").exists():
            subprocess.run(
                ["git", "init", "-q"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

        # Sym-link the real auth.json so authentication keeps working.
        auth_link = self._codex_home / "auth.json"
        global_auth = Path.home() / ".codex" / "auth.json"
        if not auth_link.exists() and global_auth.exists():
            auth_link.symlink_to(global_auth)

        prepare_codex_workspace(self.workspace)
        self._write_mcp_config()

    def _write_mcp_config(self) -> None:
        if self.phase == SUBMISSION:
            begin_submission_mcp_phase(self.session_id)
        servers = load_session_mcp_config(
            self.session_id,
            self.scenario_name,
        )

        self._logger.log(
            "mcp_config",
            {"phase": self.phase, "servers": list(servers.keys())},
        )
        config_path = self._codex_home / "config.toml"
        config_path.write_text(_build_mcp_toml(servers), encoding="utf-8")

    # ------------------------------------------------------------------
    # Subprocess invocation
    # ------------------------------------------------------------------

    async def run(self, prompt: str) -> str:
        """Execute ``codex exec`` and return the final assistant message.

        Returns an ``"ERROR: ..."`` string on subprocess failure or timeout
        rather than raising, so the LangGraph graph can continue to the
        submission phase with a degraded report.
        """
        self._setup_workspace()

        output_file = self.workspace / f"{self.phase}_output.txt"
        output_file.unlink(missing_ok=True)

        # Forward the current environment but override CODEX_HOME so that the
        # isolated config.toml and auth symlink are picked up instead of the
        # user's global ~/.codex/ directory.
        env = {**os.environ, "CODEX_HOME": str(self._codex_home)}

        cmd = ["codex", "exec"]
        if self.reasoning_effort is not None:
            cmd += ["-c", f"model_reasoning_effort={self.reasoning_effort}"]
        cmd += [
            "-m",
            self.model,
            "-C",
            str(self.workspace),
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_file),
            "--json",
            prompt,
        ]

        self._logger.log(
            "subprocess_start",
            {"command": " ".join(cmd[:6] + ["..."]), "phase": self.phase},
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
            )
            returncode, stderr_text = await self._stream_subprocess(proc)
        except asyncio.TimeoutError:
            self._logger.log(
                "subprocess_timeout", {"phase": self.phase, "timeout_s": self.timeout}
            )
            return f"ERROR: {self.phase} phase timed out after {self.timeout}s"
        except FileNotFoundError:
            self._logger.log(
                "subprocess_error", {"error": "codex binary not found in PATH"}
            )
            return "ERROR: 'codex' not found in PATH — is Codex CLI installed?"

        if returncode != 0:
            self._logger.log(
                "subprocess_error",
                {"returncode": returncode, "stderr": stderr_text[:2000]},
            )
            if self._stream_output and stderr_text.strip():
                print(stderr_text, file=sys.stderr, flush=True)
            return (
                f"ERROR: {self.phase} phase exited with code {returncode}. "
                f"stderr: {stderr_text[:400]}"
            )

        if output_file.exists():
            result = output_file.read_text(encoding="utf-8").strip()
            self._logger.log(
                "subprocess_done", {"phase": self.phase, "output_length": len(result)}
            )
            return result

        self._logger.log("subprocess_error", {"error": "output file not created"})
        return f"ERROR: {self.phase} phase produced no output"

    async def _stream_subprocess(
        self, proc: asyncio.subprocess.Process
    ) -> tuple[int, str]:
        """Read Codex stdout line-by-line until the process exits."""
        stderr_chunks: list[bytes] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout

        async def _read_stderr() -> None:
            assert proc.stderr is not None
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        stderr_task = asyncio.create_task(_read_stderr())

        try:
            assert proc.stdout is not None
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    raise asyncio.TimeoutError

                try:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise

                if not line_bytes:
                    break

                self._handle_stdout_line(
                    line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                )
        finally:
            await stderr_task

        returncode = await proc.wait()
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        return returncode, stderr_text

    def _handle_stdout_line(self, raw: str) -> None:
        """Parse one stdout line, log it, and optionally print a summary."""
        raw = raw.strip()
        if not raw:
            return
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            if self._stream_output:
                print(raw, flush=True)
            return

        self._log_codex_event(event)

    def _log_codex_event(self, event: dict) -> None:
        event_type = event.get("type", "codex_event")
        self._logger.log(event_type, {"codex_event": event})

        item = event.get("item") or {}
        if item.get("type") == "mcp_tool_call":
            tool = str(item.get("tool", ""))
            if event_type == "item.started":
                arguments = item.get("arguments")
                self._logger.log(
                    "tool_start",
                    {
                        "tool": {"name": tool},
                        "input": json.dumps(arguments, ensure_ascii=False)
                        if arguments is not None
                        else "{}",
                    },
                )
            elif event_type == "item.completed":
                if item.get("error") is not None:
                    self._logger.log("tool_error", {"output": str(item.get("error"))})
                else:
                    result = item.get("result")
                    if isinstance(result, dict):
                        content = result.get("content")
                        if isinstance(content, list):
                            output = "\n".join(
                                str(block.get("text", ""))
                                for block in content
                                if isinstance(block, dict)
                                and block.get("type") == "text"
                            )
                        else:
                            output = json.dumps(result, ensure_ascii=False)
                    else:
                        output = str(result or "")
                    self._logger.log(
                        "tool_end",
                        {"output": output, "output_type": "tool_result"},
                    )

        if self._stream_output:
            display = format_codex_event(event)
            if display:
                print(display, flush=True)

    def _forward_jsonl_events(self, text: str) -> None:
        """Parse ``codex --json`` JSONL lines and forward them to messages.jsonl."""
        for raw in text.splitlines():
            self._handle_stdout_line(raw)
