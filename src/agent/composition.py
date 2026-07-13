"""Agent composition config for optional learning modules.

This module is the boundary between benchmark-facing options and agent-side
extensions. NIKA workflows can keep passing simple flags, while the registry
works with typed extension config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LANGGRAPH_DIAGNOSIS_AGENT_TYPES = frozenset(
    {"react", "byo.langgraph", "plan-execute", "reflexion"}
)


@dataclass(frozen=True)
class ToolRefinementConfig:
    enabled: bool = False
    library_id: str = "default"
    tool_doc_chars: int = 500
    convergence_threshold: float = 0.75


@dataclass(frozen=True)
class ProceduralMemoryConfig:
    mode: str = "off"
    bank: str = "default"
    top_k: int = 5
    token_budget: int = 1500
    max_skill_age: int = 8
    pool_size: int = 32
    evolution_threshold: int = 6
    best_of_n: int = 3
    ppo_epsilon: float = 0.2
    selection_epsilon: float = 0.3

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class AgentRunConfig:
    agent_type: str
    llm_provider: str
    model: str
    max_steps: int
    max_attempts: int = 3
    session_id: str = ""
    tool_refinement: ToolRefinementConfig = field(default_factory=ToolRefinementConfig)
    procedural_memory: ProceduralMemoryConfig = field(
        default_factory=ProceduralMemoryConfig
    )

    @property
    def normalized_agent_type(self) -> str:
        normalized = self.agent_type.strip().lower().replace("_", "-")
        if normalized == "plan-and-execute":
            return "plan-execute"
        return normalized

    @property
    def llm_backend(self) -> str:
        """Compatibility name used by persisted pre-upstream run metadata."""
        return self.llm_provider

    @property
    def extensions_enabled(self) -> bool:
        return self.tool_refinement.enabled or self.procedural_memory.enabled


def validate_agent_extensions(config: AgentRunConfig) -> None:
    """Validate extension compatibility without requiring a live session."""
    agent_type = config.normalized_agent_type
    if agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES:
        supported = ", ".join(sorted(LANGGRAPH_DIAGNOSIS_AGENT_TYPES))
        raise ValueError(f"unsupported local workflow; choose one of: {supported}")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if (
        config.tool_refinement.enabled
        and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES
    ):
        raise ValueError(
            "Tool Refinement supports only the NIKA ReAct diagnosis workflow."
        )
    if config.tool_refinement.tool_doc_chars < 100:
        raise ValueError("Tool Refinement documentation size must be >= 100")
    if not 0 <= config.tool_refinement.convergence_threshold <= 1:
        raise ValueError("Tool Refinement convergence threshold must be in [0, 1]")
    if config.procedural_memory.mode not in {"off", "read", "evolve"}:
        raise ValueError("Procedural Memory mode must be one of: off, read, evolve")
    if config.procedural_memory.top_k < 1:
        raise ValueError("Procedural Memory top-k must be >= 1")
    if config.procedural_memory.token_budget < 100:
        raise ValueError("Procedural Memory token budget must be >= 100")
    if config.procedural_memory.max_skill_age < 1:
        raise ValueError("Procedural Memory maximum skill age must be >= 1")
    if config.procedural_memory.pool_size < 1:
        raise ValueError("Procedural Memory pool size must be >= 1")
    if config.procedural_memory.evolution_threshold < 1:
        raise ValueError("Procedural Memory update threshold must be >= 1")
    if config.procedural_memory.best_of_n < 1:
        raise ValueError("Procedural Memory best-of-N must be >= 1")
    if not 0 <= config.procedural_memory.ppo_epsilon <= 1:
        raise ValueError("Procedural Memory PPO epsilon must be in [0, 1]")
    if not 0 <= config.procedural_memory.selection_epsilon <= 1:
        raise ValueError("Procedural Memory selection epsilon must be in [0, 1]")
    if (
        config.procedural_memory.enabled
        and agent_type not in LANGGRAPH_DIAGNOSIS_AGENT_TYPES
    ):
        supported = ", ".join(sorted(LANGGRAPH_DIAGNOSIS_AGENT_TYPES))
        raise ValueError(
            f"Procedural Memory is supported only for these workflows: {supported}"
        )


def validate_agent_composition(config: AgentRunConfig) -> None:
    """Validate config before constructing an agent instance."""
    if not config.session_id:
        raise ValueError("session_id is required to construct an agent")
    validate_agent_extensions(config)
