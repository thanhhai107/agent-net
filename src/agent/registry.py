"""Agent type registry used by ``nika agent run``."""

from typing import Any

from agent.cli.agent import CliAgent
from agent.langgraph.plan_execute_agent import PlanExecuteAgent
from agent.langgraph.react_agent import BasicReActAgent
from agent.langgraph.reflexion_agent import ReflexionAgent
from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.service import HybridMemoryModule
from agent.mock.mock_agent import MockAgent

MEMORY_COMPATIBLE_AGENT_TYPES = frozenset({"react", "plan-execute", "reflexion"})


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
    memory_mode: str = "off",
    memory_bank: str = "default",
    memory_top_k: int = 5,
    memory_token_budget: int = 1500,
) -> Any:
    """Instantiate an agent for ``agent_type``."""
    normalized_type = agent_type.lower()
    if tool_evolution_enabled and normalized_type not in MEMORY_COMPATIBLE_AGENT_TYPES:
        raise ValueError(
            "Tool Evolution supports react, plan-execute, and reflexion workflows."
        )
    if memory_mode not in {"off", "read", "evolve"}:
        raise ValueError("memory_mode must be one of: off, read, evolve")
    if memory_mode != "off" and normalized_type not in MEMORY_COMPATIBLE_AGENT_TYPES:
        supported = ", ".join(sorted(MEMORY_COMPATIBLE_AGENT_TYPES))
        raise ValueError(f"memory is supported only for these workflows: {supported}")

    use_memory = memory_mode != "off"
    match normalized_type:
        case "react":
            kwargs = dict(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                oracle_routing=oracle_routing,
                tool_evolution_enabled=tool_evolution_enabled,
                tool_library_id=tool_library_id,
                tool_evolution_mode=tool_evolution_mode,
            )
            if use_memory:
                kwargs["use_problem_tool_hints"] = False
            agent = BasicReActAgent(**kwargs)
        case "plan-execute":
            kwargs = dict(
                session_id=session_id,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                oracle_routing=oracle_routing,
                tool_evolution_enabled=tool_evolution_enabled,
                tool_library_id=tool_library_id,
                tool_evolution_mode=tool_evolution_mode,
            )
            if use_memory:
                kwargs["use_problem_tool_hints"] = False
            agent = PlanExecuteAgent(**kwargs)
        case "reflexion":
            kwargs = dict(
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
            if use_memory:
                kwargs["use_problem_tool_hints"] = False
            agent = ReflexionAgent(**kwargs)
        case "mock":
            agent = MockAgent(
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
            agent = CliAgent(
                session_id=session_id,
                model=model,
                reasoning_effort=reasoning_effort,
                oracle_routing=oracle_routing,
                stream_output=stream_output,
            )
        case _:
            raise ValueError(f"Unsupported agent type: {agent_type!r}")

    if not use_memory:
        return agent
    return MemoryAugmentedAgent(
        agent,
        HybridMemoryModule(
            bank_id=memory_bank,
            llm_backend=llm_backend,
            model=model,
        ),
        memory_mode=memory_mode,
        memory_top_k=memory_top_k,
        memory_token_budget=memory_token_budget,
    )
