"""Agent type registry used by ``nika agent run``."""

from typing import Any

from agent.cli.agent import CliAgent
from agent.composition import (
    AgentRunConfig,
    validate_agent_composition,
    workflow_agent_kwargs,
    wrap_agent_extensions,
)
from agent.langgraph.plan_execute_agent import PlanExecuteAgent
from agent.langgraph.react_agent import BasicReActAgent
from agent.langgraph.reflexion_agent import ReflexionAgent
from agent.mock.mock_agent import MockAgent


def create_agent(config: AgentRunConfig) -> Any:
    """Instantiate an agent for ``agent_type``."""
    validate_agent_composition(config)

    normalized_type = config.normalized_agent_type
    match normalized_type:
        case "react":
            agent = BasicReActAgent(**workflow_agent_kwargs(config))
        case "plan-execute":
            agent = PlanExecuteAgent(**workflow_agent_kwargs(config))
        case "reflexion":
            agent = ReflexionAgent(**workflow_agent_kwargs(config, reflexion=True))
        case "mock":
            agent = MockAgent(
                session_id=config.session_id,
                llm_backend=config.llm_backend,
                model=config.model,
                max_steps=config.max_steps,
            )
        case "sdk":
            raise ValueError(
                "Agent type 'sdk' is not implemented yet. "
                "See docs/README.md for the current agent boundary."
            )
        case "cli":
            agent = CliAgent(
                session_id=config.session_id,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                stream_output=config.stream_output,
            )
        case _:
            raise ValueError(f"Unsupported agent type: {config.agent_type!r}")

    return wrap_agent_extensions(agent, config)
