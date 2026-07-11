"""Agent composition config for optional learning modules.

This module is the boundary between benchmark-facing options and agent-side
extensions. NIKA workflows can keep passing simple flags, while the registry
works with typed extension config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
LANGGRAPH_DIAGNOSIS_AGENT_TYPES = frozenset({"react", "byo.langgraph"})


@dataclass(frozen=True)
class ToolEvolutionConfig:
    enabled: bool = False
    library_id: str = "default"
    tool_doc_chars: int = 500
    convergence_threshold: float = 0.75


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "off"
    bank: str = "default"
    top_k: int = 5
    token_budget: int = 1500
    max_skill_age: int = 4
    pool_size: int = 32
    evolution_threshold: int = 3
    best_of_n: int = 3
    ppo_epsilon: float = 0.2

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class AgentRunConfig:
    agent_type: str
    llm_provider: str
    model: str
    max_steps: int
    session_id: str = ""
    tool_evolution: ToolEvolutionConfig = field(default_factory=ToolEvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def normalized_agent_type(self) -> str:
        return self.agent_type.lower()

    @property
    def llm_backend(self) -> str:
        """Compatibility name used by persisted pre-upstream run metadata."""
        return self.llm_provider

    @property
    def extensions_enabled(self) -> bool:
        return self.tool_evolution.enabled or self.memory.enabled


def validate_agent_extensions(config: AgentRunConfig) -> None:
    """Validate extension compatibility without requiring a live session."""
    agent_type = config.normalized_agent_type
    if (
        config.tool_evolution.enabled
        and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES
    ):
        raise ValueError(
            "Tool Evolution supports only the NIKA ReAct diagnosis workflow."
        )
    if config.tool_evolution.tool_doc_chars < 100:
        raise ValueError("tool_evolution tool_doc_chars must be >= 100")
    if not 0 <= config.tool_evolution.convergence_threshold <= 1:
        raise ValueError("tool_evolution convergence_threshold must be in [0, 1]")
    if config.memory.mode not in {"off", "read", "evolve"}:
        raise ValueError("memory_mode must be one of: off, read, evolve")
    if config.memory.top_k < 1:
        raise ValueError("memory top_k must be >= 1")
    if config.memory.token_budget < 100:
        raise ValueError("memory token_budget must be >= 100")
    if config.memory.max_skill_age < 1:
        raise ValueError("memory max_skill_age must be >= 1")
    if config.memory.pool_size < 1:
        raise ValueError("memory pool_size must be >= 1")
    if config.memory.evolution_threshold < 1:
        raise ValueError("memory evolution_threshold must be >= 1")
    if config.memory.best_of_n < 1:
        raise ValueError("memory best_of_n must be >= 1")
    if config.memory.ppo_epsilon < 0:
        raise ValueError("memory ppo_epsilon must be >= 0")
    if config.memory.enabled and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES:
        supported = ", ".join(sorted(LANGGRAPH_DIAGNOSIS_AGENT_TYPES))
        raise ValueError(f"memory is supported only for these workflows: {supported}")


def validate_agent_composition(config: AgentRunConfig) -> None:
    """Validate config before constructing an agent instance."""
    if not config.session_id:
        raise ValueError("session_id is required to construct an agent")
    validate_agent_extensions(config)

