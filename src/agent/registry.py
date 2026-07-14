"""Agent type registry used by ``nika agent run``."""

from typing import Any


def create_agent(
    agent_type: str,
    *,
    session_id: str,
    model: str,
    llm_provider: str | None = None,
    max_steps: int = 20,
    reasoning_effort: str | None = None,
    stream_output: bool = True,
) -> Any:
    """Instantiate an agent for ``agent_type``."""
    match agent_type.lower():
        case "react" | "byo.langgraph":
            from agent.byo.langgraph.react_agent import BasicReActAgent

            if not llm_provider:
                raise ValueError(
                    "react agent requires an LLM provider: set NIKA_LLM_PROVIDER in .env or pass -p/--provider."
                )
            return BasicReActAgent(
                session_id=session_id,
                llm_provider=llm_provider,
                model=model,
                max_steps=max_steps,
            )
        case "mock":
            from agent.mock.mock_agent import MockAgent

            return MockAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
            )
        case "sdk.claude_sdk":
            from agent.sdk.claude_sdk.agent import ClaudeSdkAgent

            return ClaudeSdkAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
                stream_output=stream_output,
            )
        case "sdk.codex_sdk":
            from agent.sdk.codex_sdk.agent import CodexSdkAgent

            return CodexSdkAgent(
                session_id=session_id,
                model=model,
                reasoning_effort=reasoning_effort,
                stream_output=stream_output,
            )
        case "local_cli.codex_cli":
            from agent.local_cli.codex_cli.agent import CodexCliAgent

            return CodexCliAgent(
                session_id=session_id,
                model=model,
                reasoning_effort=reasoning_effort,
                stream_output=stream_output,
            )
        case "local_cli.claude_cli":
            from agent.local_cli.claude_cli.agent import ClaudeAgent

            return ClaudeAgent(
                session_id=session_id,
                model=model,
                stream_output=stream_output,
            )
        case "byo.mcp_agent":
            from agent.byo.mcp_agent.agent import McpAgent

            return McpAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
                stream_output=stream_output,
            )
        case "byo.autogen":
            from agent.byo.autogen.agent import AutogenAgent

            return AutogenAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
                stream_output=stream_output,
            )
        case "community.sade":
            from agent.community.sade.agent import SadeAgent

            return SadeAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
            )
        case _:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")
