"""Agent composition config for optional learning modules.

This module is the boundary between benchmark-facing options and agent-side
extensions. NIKA workflows can keep passing simple flags, while the registry
works with typed extension config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.module_config import module_defaults

LANGGRAPH_DIAGNOSIS_AGENT_TYPES = frozenset({"react", "plan-execute", "reflexion"})


@dataclass(frozen=True)
class ToolRefinementConfig:
    enabled: bool = False
    library_id: str = "default"
    learning_mode: str = "evolve"
    update_due: bool = True
    tool_doc_chars: int = field(
        default_factory=lambda: module_defaults().tool_refinement.tool_doc_chars
    )
    convergence_threshold: float = field(
        default_factory=lambda: module_defaults().tool_refinement.convergence_threshold
    )
    exploration_similarity_threshold: float = field(
        default_factory=lambda: (
            module_defaults().tool_refinement.exploration_similarity_threshold
        )
    )
    explorer_reflection_limit: int = field(
        default_factory=lambda: (
            module_defaults().tool_refinement.explorer_reflection_limit
        )
    )
    update_interval: int = field(
        default_factory=lambda: module_defaults().tool_refinement.update_interval
    )
    min_new_trials: int = field(
        default_factory=lambda: module_defaults().tool_refinement.min_new_trials
    )
    max_tools_per_update: int = field(
        default_factory=lambda: module_defaults().tool_refinement.max_tools_per_update
    )
    publish_min_utility: float = field(
        default_factory=lambda: module_defaults().tool_refinement.publish_min_utility
    )
    explorer_model: str = ""
    analyzer_model: str = ""
    rewriter_model: str = ""


@dataclass(frozen=True)
class ProceduralMemoryConfig:
    mode: str = "off"
    bank: str = "default"
    token_budget: int = field(
        default_factory=lambda: module_defaults().procedural_memory.token_budget
    )
    max_skill_age: int = field(
        default_factory=lambda: module_defaults().procedural_memory.max_skill_age
    )
    pool_size: int = field(
        default_factory=lambda: module_defaults().procedural_memory.pool_size
    )
    evolution_threshold: int = field(
        default_factory=lambda: module_defaults().procedural_memory.evolution_threshold
    )
    best_of_n: int = field(
        default_factory=lambda: module_defaults().procedural_memory.best_of_n
    )
    ppo_epsilon: float = field(
        default_factory=lambda: module_defaults().procedural_memory.ppo_epsilon
    )
    selection_epsilon: float = field(
        default_factory=lambda: module_defaults().procedural_memory.selection_epsilon
    )
    experience_pool_size: int = field(
        default_factory=lambda: module_defaults().procedural_memory.experience_pool_size
    )
    baseline_ema_alpha: float = field(
        default_factory=lambda: module_defaults().procedural_memory.baseline_ema_alpha
    )
    selection_epsilon_decay_cases: int = field(
        default_factory=lambda: (
            module_defaults().procedural_memory.selection_epsilon_decay_cases
        )
    )
    acceptance_margin: float = field(
        default_factory=lambda: module_defaults().procedural_memory.acceptance_margin
    )
    verifier: str = field(
        default_factory=lambda: module_defaults().procedural_memory.verifier
    )
    holdout_size: int = field(
        default_factory=lambda: module_defaults().procedural_memory.holdout_size
    )
    min_positive_advantage: int = field(
        default_factory=lambda: (
            module_defaults().procedural_memory.min_positive_advantage
        )
    )
    evolver_model: str = ""
    policy_scorer_model: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class AgentRunConfig:
    agent_type: str
    llm_provider: str
    model: str
    max_steps: int
    max_attempts: int = field(
        default_factory=lambda: module_defaults().baseline.max_attempts
    )
    session_id: str = ""
    tool_refinement: ToolRefinementConfig = field(default_factory=ToolRefinementConfig)
    procedural_memory: ProceduralMemoryConfig = field(
        default_factory=ProceduralMemoryConfig
    )

    @property
    def normalized_agent_type(self) -> str:
        normalized = self.agent_type.strip().lower().replace("_", "-")
        if normalized == "byo.langgraph":
            return "react"
        if normalized == "plan-and-execute":
            return "plan-execute"
        return normalized

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
    if not 0 <= config.tool_refinement.exploration_similarity_threshold <= 1:
        raise ValueError("Tool Refinement exploration similarity must be in [0, 1]")
    if config.tool_refinement.explorer_reflection_limit < 0:
        raise ValueError("Tool Refinement reflection limit must be >= 0")
    if config.tool_refinement.learning_mode not in {"evolve", "read"}:
        raise ValueError("Tool Refinement learning mode must be evolve or read")
    if (
        min(
            config.tool_refinement.update_interval,
            config.tool_refinement.min_new_trials,
            config.tool_refinement.max_tools_per_update,
        )
        < 1
    ):
        raise ValueError("Tool Refinement update controls must be >= 1")
    if not 0 <= config.tool_refinement.publish_min_utility <= 1:
        raise ValueError("Tool Refinement publication utility must be in [0, 1]")
    if config.procedural_memory.mode not in {"off", "read", "evolve"}:
        raise ValueError("Procedural Memory mode must be one of: off, read, evolve")
    if config.procedural_memory.token_budget < 100:
        raise ValueError("Procedural Memory token budget must be >= 100")
    if config.procedural_memory.max_skill_age < 1:
        raise ValueError("Procedural Memory maximum skill age must be >= 1")
    if config.procedural_memory.pool_size < 1:
        raise ValueError("Procedural Memory pool size must be >= 1")
    if config.procedural_memory.evolution_threshold < 2:
        raise ValueError("Procedural Memory evolution batch size must be >= 2")
    if config.procedural_memory.best_of_n < 1:
        raise ValueError("Procedural Memory best-of-N must be >= 1")
    if not 0 <= config.procedural_memory.ppo_epsilon <= 1:
        raise ValueError("Procedural Memory PPO epsilon must be in [0, 1]")
    if not 0 <= config.procedural_memory.selection_epsilon <= 1:
        raise ValueError("Procedural Memory selection epsilon must be in [0, 1]")
    if config.procedural_memory.experience_pool_size < 1:
        raise ValueError("Procedural Memory experience pool size must be >= 1")
    if not 0 < config.procedural_memory.baseline_ema_alpha <= 1:
        raise ValueError("Procedural Memory baseline EMA alpha must be in (0, 1]")
    if config.procedural_memory.selection_epsilon_decay_cases < 1:
        raise ValueError("Procedural Memory epsilon decay cases must be >= 1")
    if config.procedural_memory.acceptance_margin < 0:
        raise ValueError("Procedural Memory acceptance margin must be >= 0")
    if config.procedural_memory.verifier not in {
        "behavioral_replay",
        "structured_replay",
        "policy_logprob",
    }:
        raise ValueError("Procedural Memory verifier is invalid")
    if config.procedural_memory.holdout_size < 1:
        raise ValueError("Procedural Memory holdout size must be >= 1")
    if config.procedural_memory.min_positive_advantage < 0:
        raise ValueError("Procedural Memory positive-advantage support must be >= 0")
    if (
        config.procedural_memory.min_positive_advantage
        > config.procedural_memory.holdout_size
    ):
        raise ValueError(
            "Procedural Memory positive-advantage support exceeds holdout size"
        )
    if (
        config.procedural_memory.holdout_size
        >= config.procedural_memory.evolution_threshold
    ):
        raise ValueError("Procedural Memory holdout must leave a generation trajectory")
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
