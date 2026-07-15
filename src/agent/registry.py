"""Agent type registry used by ``nika agent run``."""

from typing import Any


def create_agent(
    agent_type: str,
    *,
    session_id: str,
    model: str,
    llm_provider: str | None = None,
    max_steps: int = 20,
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
        case _:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")
