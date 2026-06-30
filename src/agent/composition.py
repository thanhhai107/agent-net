"""Agent composition config for optional learning modules.

This module is the boundary between benchmark-facing options and agent-side
extensions. NIKA workflows can keep passing simple flags, while the registry
works with typed extension config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.defaults import DEFAULT_MAX_STEPS
from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.service import ProceduralMemoryModule

LANGGRAPH_DIAGNOSIS_AGENT_TYPES = frozenset({"react", "plan-execute", "reflexion"})


@dataclass(frozen=True)
class ToolEvolutionConfig:
    enabled: bool = False
    library_id: str = "default"
    mode: str = "dual"


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "off"
    bank: str = "default"
    top_k: int = 5
    token_budget: int = 1500

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class HarnessConfig:
    target_agent_path: str | None = None

    @property
    def enabled(self) -> bool:
        return self.target_agent_path is not None


@dataclass(frozen=True)
class AgentRunConfig:
    agent_type: str
    llm_backend: str
    model: str
    session_id: str = ""
    max_steps: int = DEFAULT_MAX_STEPS
    max_attempts: int = 3
    reasoning_effort: str | None = None
    stream_output: bool = True
    oracle_routing: bool = False
    tool_evolution: ToolEvolutionConfig = field(default_factory=ToolEvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def normalized_agent_type(self) -> str:
        return self.agent_type.lower()


def validate_agent_extensions(config: AgentRunConfig) -> None:
    """Validate extension compatibility without requiring a live session."""
    agent_type = config.normalized_agent_type
    if (
        config.tool_evolution.enabled
        and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES
    ):
        raise ValueError(
            "Tool Evolution supports react, plan-execute, and reflexion workflows."
        )
    if config.memory.mode not in {"off", "read", "evolve"}:
        raise ValueError("memory_mode must be one of: off, read, evolve")
    if config.memory.enabled and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES:
        supported = ", ".join(sorted(LANGGRAPH_DIAGNOSIS_AGENT_TYPES))
        raise ValueError(f"memory is supported only for these workflows: {supported}")


def validate_agent_composition(config: AgentRunConfig) -> None:
    """Validate config before constructing an agent instance."""
    if not config.session_id:
        raise ValueError("session_id is required to construct an agent")
    validate_agent_extensions(config)


def workflow_agent_kwargs(
    config: AgentRunConfig,
    *,
    reflexion: bool = False,
) -> dict[str, Any]:
    """Build constructor kwargs for LangGraph diagnosis workflows."""
    kwargs = {
        "session_id": config.session_id,
        "llm_backend": config.llm_backend,
        "model": config.model,
        "max_steps": config.max_steps,
        "oracle_routing": config.oracle_routing,
        "tool_evolution_enabled": config.tool_evolution.enabled,
        "tool_library_id": config.tool_evolution.library_id,
        "tool_evolution_mode": config.tool_evolution.mode,
    }
    if reflexion:
        kwargs["max_attempts"] = config.max_attempts
    if config.memory.enabled:
        kwargs["use_problem_tool_hints"] = False
    return kwargs


def wrap_agent_extensions(agent: Any, config: AgentRunConfig) -> Any:
    """Wrap a constructed agent with agent-side extensions that need wrappers."""
    if not config.memory.enabled:
        return agent
    return MemoryAugmentedAgent(
        agent,
        ProceduralMemoryModule(
            bank_id=config.memory.bank,
            llm_backend=config.llm_backend,
            model=config.model,
        ),
        memory_mode=config.memory.mode,
        memory_top_k=config.memory.top_k,
        memory_token_budget=config.memory.token_budget,
    )
