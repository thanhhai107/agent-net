"""Claude Agent SDK worker — one phase per ClaudeSDKClient session."""

from __future__ import annotations

from typing import Any

from agent.sdk.claude_sdk.config import prepare_claude_sdk_env
from agent.sdk.mcp import to_sdk_mcp_servers
from agent.utils.loggers import MessageLogger
from agent.utils.mcp_servers import MCPServerConfig, select_diagnosis_servers
from agent.utils.phases import PHASES, SUBMISSION
from nika.utils.logger import system_logger


def _normalize_tool_name(name: str) -> str:
    """Map claude-agent-sdk MCP names (``mcp__server__tool``) to short tool ids."""
    prefix = "mcp__"
    if name.startswith(prefix):
        remainder = name[len(prefix) :]
        if "__" in remainder:
            return remainder.split("__", 1)[1]
    return name


def _usage_metadata(usage: dict[str, Any] | None) -> dict[str, int]:
    u = usage or {}
    return {
        "input_tokens": (
            u.get("input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
        ),
        "output_tokens": u.get("output_tokens", 0),
    }


class ClaudeSdkWorker:
    """Drive one troubleshooting phase via ``claude-agent-sdk``."""

    def __init__(
        self,
        session_id: str,
        session_dir: str,
        phase: str,
        model: str,
        max_steps: int = 20,
        scenario_name: str = "",
        problem_names: list[str] | None = None,
        *,
        system_prompt: str,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

        self.session_id = session_id
        self.session_dir = session_dir
        self.phase = phase
        self.model = model
        self.max_steps = max_steps
        self.scenario_name = scenario_name
        self.problem_names = problem_names or []
        self.system_prompt = system_prompt
        self._logger = MessageLogger(agent=phase, session_dir=session_dir)

    def _load_mcp_servers(self) -> dict[str, Any]:
        mcp = MCPServerConfig(session_id=self.session_id)
        if self.phase == SUBMISSION:
            servers = mcp.load_config(if_submit=True)
        else:
            server_names = select_diagnosis_servers(self.scenario_name, self.problem_names)
            servers = mcp.load_filtered_config(server_names)
        return to_sdk_mcp_servers(servers)

    async def run(self, prompt: str) -> str:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ClaudeSDKClient,
                ResultMessage,
                SystemMessage,
                TextBlock,
                ThinkingBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
            )
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is not installed. Run: uv sync --extra sdk"
            ) from exc

        mcp_servers = self._load_mcp_servers()
        sdk_env = prepare_claude_sdk_env(session_id=self.session_id)

        self._logger.log(
            "mcp_config",
            {"phase": self.phase, "servers": list(mcp_servers.keys())},
        )
        self._logger.log(
            "llm_start",
            {
                "messages": {"role": "user", "content": prompt[:500]},
                "model": {"name": self.model},
                "mcp_servers": list(mcp_servers.keys()),
            },
        )

        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            model=self.model,
            mcp_servers=mcp_servers,
            max_turns=self.max_steps,
            permission_mode="bypassPermissions",
            env=sdk_env,
        )

        result_text = ""
        turn_text: list[str] = []

        def _flush_turn() -> None:
            nonlocal turn_text
            if turn_text:
                self._logger.log("llm_end", {"text": "\n".join(turn_text), "usage_metadata": {}})
                turn_text = []

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_messages():
                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        system_logger.info(
                            f"claude_sdk/{self.phase}: session started - "
                            f"{message.data.get('session_id')}"
                        )
                    elif isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, (ThinkingBlock, TextBlock)):
                                text = block.thinking if isinstance(block, ThinkingBlock) else block.text
                                turn_text.append(text)
                            elif isinstance(block, ToolUseBlock):
                                _flush_turn()
                                self._logger.log(
                                    "tool_start",
                                    {
                                        "tool": {"name": _normalize_tool_name(block.name)},
                                        "input": str(block.input),
                                    },
                                )
                    elif isinstance(message, UserMessage):
                        _flush_turn()
                        content = message.content if isinstance(message.content, list) else []
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                if block.is_error:
                                    self._logger.log("tool_error", {"output": str(block.content)})
                                else:
                                    self._logger.log(
                                        "tool_end",
                                        {"output": str(block.content), "output_type": "tool_result"},
                                    )
                    elif isinstance(message, ResultMessage):
                        _flush_turn()
                        result_text = message.result or ""
                        md = _usage_metadata(message.usage)
                        self._logger.log("llm_end", {"text": result_text, "usage_metadata": md})
                        system_logger.info(
                            f"claude_sdk/{self.phase}: complete - "
                            f"stop_reason={message.stop_reason}"
                        )
                        break
        except Exception as exc:
            self._logger.log("agent_error", {"phase": self.phase, "error": str(exc)})
            return f"ERROR: {exc}"

        self._logger.log("agent_done", {"phase": self.phase, "report_length": len(result_text)})
        return result_text
