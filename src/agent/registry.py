"""Agent type registry used by ``nika agent run``."""

from typing import Any

from agent.cli.agent import CliAgent
from agent.langgraph.plan_execute_agent import PlanExecuteAgent
from agent.langgraph.react_agent import BasicReActAgent
from agent.langgraph.reflexion_agent import ReflexionAgent
from agent.mock.mock_agent import MockAgent


def create_agent(
    agent_type: str,
    *,
    session_id: str,
    llm_backend: str,
    model: str,
    max_steps: int = 20,
    max_attempts: int = 3,
    reasoning_effort: str | None = None,
    stream_output: bool = True,
    oracle_routing: bool = False,
    tool_evolution_enabled: bool = False,
    tool_library_id: str = "default",
    tool_evolution_mode: str = "dual",
) -> Any:
    """Instantiate an agent for ``agent_type``."""
    normalized_type = agent_type.lower()
    if tool_evolution_enabled and normalized_type not in {
        "react",
        "plan-execute",
        "reflexion",
    }:
        raise ValueError(
            "Tool Evolution supports react, plan-execute, and reflexion workflows."
        )

    match normalized_type:
        case "react":
            return BasicReActAgent(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                oracle_routing=oracle_routing,
                tool_evolution_enabled=tool_evolution_enabled,
                tool_library_id=tool_library_id,
                tool_evolution_mode=tool_evolution_mode,
            )
        case "plan-execute":
            return PlanExecuteAgent(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                oracle_routing=oracle_routing,
                tool_evolution_enabled=tool_evolution_enabled,
                tool_library_id=tool_library_id,
                tool_evolution_mode=tool_evolution_mode,
            )
        case "reflexion":
            return ReflexionAgent(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                max_attempts=max_attempts,
                oracle_routing=oracle_routing,
                tool_evolution_enabled=tool_evolution_enabled,
                tool_library_id=tool_library_id,
                tool_evolution_mode=tool_evolution_mode,
            )
        case "mock":
            return MockAgent(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
            )
        case "sdk":
            raise ValueError(
                "Agent type 'sdk' is not implemented yet. "
                "See src/agent/README.md for the Claude/Codex SDK path."
            )
        case "cli":
            return CliAgent(
                session_id=session_id,
                model=model,
                reasoning_effort=reasoning_effort,
                oracle_routing=oracle_routing,
                stream_output=stream_output,
            )
        case _:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")
