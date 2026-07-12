"""Factory for local workflows that sit outside NIKA's upstream registry."""

from agent.composition import AgentRunConfig, validate_agent_composition
from agent.extensions.plan_execute_agent import PlanExecuteAgent
from agent.extensions.react_agent import create_react_agent
from agent.extensions.reflexion_agent import ReflexionAgent


def create_extension_agent(config: AgentRunConfig):
    validate_agent_composition(config)
    factories = {
        "react": create_react_agent,
        "byo.langgraph": create_react_agent,
        "plan-execute": PlanExecuteAgent,
        "reflexion": ReflexionAgent,
    }
    return factories[config.normalized_agent_type](config)
