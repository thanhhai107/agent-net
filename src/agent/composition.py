"""Agent composition config for optional learning modules.

This module is the boundary between benchmark-facing options and agent-side
extensions. NIKA workflows can keep passing simple flags, while the registry
works with typed extension config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.memory.adapter import MemoryAugmentedAgent
from agent.memory.service import ProceduralMemoryModule

LANGGRAPH_DIAGNOSIS_AGENT_TYPES = frozenset({"react", "plan-execute", "reflexion"})


@dataclass(frozen=True)
class ToolEvolutionConfig:
    enabled: bool = False
    library_id: str = "default"
    tool_doc_chars: int = 500
    prompt_doc_limit: int = 6
    scoped_prompt_doc_limit: int = 4
    planned_checks: int = 4
    next_checks: int = 2
    convergence_threshold: float = 0.75


@dataclass(frozen=True)
class MemoryConfig:
    mode: str = "off"
    bank: str = "default"
    top_k: int = 5
    token_budget: int = 1500
    skill_selector_mode: str = "lcb"
    meta_controller_mode: str = "heuristic"
    max_skill_age: int = 4
    selector_min_lcb: float = -0.05
    selector_nominee_k: int = 3
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
    llm_backend: str
    model: str
    max_steps: int
    session_id: str = ""
    max_attempts: int = 3
    stream_output: bool = True
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
    if config.tool_evolution.tool_doc_chars < 100:
        raise ValueError("tool_evolution tool_doc_chars must be >= 100")
    if config.tool_evolution.prompt_doc_limit < 1:
        raise ValueError("tool_evolution prompt_doc_limit must be >= 1")
    if config.tool_evolution.scoped_prompt_doc_limit < 1:
        raise ValueError("tool_evolution scoped_prompt_doc_limit must be >= 1")
    if config.tool_evolution.planned_checks < 0:
        raise ValueError("tool_evolution planned_checks must be >= 0")
    if config.tool_evolution.next_checks < 0:
        raise ValueError("tool_evolution next_checks must be >= 0")
    if not 0 <= config.tool_evolution.convergence_threshold <= 1:
        raise ValueError("tool_evolution convergence_threshold must be in [0, 1]")
    if config.memory.mode not in {"off", "read", "evolve"}:
        raise ValueError("memory_mode must be one of: off, read, evolve")
    if config.memory.skill_selector_mode not in {"lcb", "llm_topk_lcb"}:
        raise ValueError("memory skill selector must be one of: lcb, llm_topk_lcb")
    if config.memory.meta_controller_mode not in {"heuristic", "llm"}:
        raise ValueError("memory meta controller must be one of: heuristic, llm")
    if config.memory.top_k < 1:
        raise ValueError("memory top_k must be >= 1")
    if config.memory.token_budget < 100:
        raise ValueError("memory token_budget must be >= 100")
    if config.memory.max_skill_age < 1:
        raise ValueError("memory max_skill_age must be >= 1")
    if config.memory.selector_nominee_k < 1:
        raise ValueError("memory selector_nominee_k must be >= 1")
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
        "tool_evolution_enabled": config.tool_evolution.enabled,
        "tool_library_id": config.tool_evolution.library_id,
        "tool_doc_chars": config.tool_evolution.tool_doc_chars,
        "tool_prompt_doc_limit": config.tool_evolution.prompt_doc_limit,
        "tool_scoped_prompt_doc_limit": config.tool_evolution.scoped_prompt_doc_limit,
        "tool_planned_checks": config.tool_evolution.planned_checks,
        "tool_next_checks": config.tool_evolution.next_checks,
    }
    if reflexion:
        kwargs["max_attempts"] = config.max_attempts
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
            pool_size=config.memory.pool_size,
            evolution_threshold=config.memory.evolution_threshold,
            best_of_n=config.memory.best_of_n,
            ppo_epsilon=config.memory.ppo_epsilon,
        ),
        memory_mode=config.memory.mode,
        memory_top_k=config.memory.top_k,
        memory_token_budget=config.memory.token_budget,
        memory_skill_selector_mode=config.memory.skill_selector_mode,
        memory_meta_controller_mode=config.memory.meta_controller_mode,
        memory_max_skill_age=config.memory.max_skill_age,
        memory_selector_min_lcb=config.memory.selector_min_lcb,
        memory_selector_nominee_k=config.memory.selector_nominee_k,
    )
