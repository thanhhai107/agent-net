"""Agent type registry used by ``nika agent run``."""

from typing import Any

from agent.codex_cli.agent import CodexCliAgent
from agent.claude_cli.agent import ClaudeAgent
from agent.langgraph.react_agent import BasicReActAgent
from agent.mock.mock_agent import MockAgent


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
        case "react":
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
            return MockAgent(
                session_id=session_id,
                model=model,
                max_steps=max_steps,
            )
        case "sdk":
            raise ValueError(
                "Agent type 'sdk' is not implemented yet. "
                "See src/agent/README.md for the codex_sdk / claude_sdk path."
            )
        case "codex_cli":
            return CodexCliAgent(
                session_id=session_id,
                model=model,
                reasoning_effort=reasoning_effort,
                stream_output=stream_output,
            )
        case "claude_cli":
            return ClaudeAgent(
                session_id=session_id,
                model=model,
                stream_output=stream_output,
            )
        case _:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")
