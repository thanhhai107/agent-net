"""Execution boundary for optional learning extensions."""

from __future__ import annotations

import asyncio

from agent.composition import AgentRunConfig, validate_agent_composition
from agent.extensions.factory import create_extension_agent
from nika.service.mcp_gateway.lifecycle import mcp_gateway_for_session
from nika.utils.logger import bind_session_dir, log_event
from nika.utils.session import Session
from nika.workflows.agent.run import start_agent as start_nika_agent


def _write_extension_metadata(session: Session, config: AgentRunConfig) -> None:
    session.update_session(
        "procedural_memory_enabled", config.procedural_memory.enabled
    )
    session.update_session("allow_learning_updates", config.allow_learning_updates)
    session.update_session("procedural_memory_bank", config.procedural_memory.bank)
    session.update_session(
        "procedural_memory_store_path",
        str(config.procedural_memory.store_path or ""),
    )
    session.update_session(
        "procedural_memory_token_budget", config.procedural_memory.token_budget
    )
    session.update_session(
        "procedural_memory_max_skill_age", config.procedural_memory.max_skill_age
    )
    session.update_session(
        "procedural_memory_pool_size", config.procedural_memory.pool_size
    )
    session.update_session(
        "procedural_memory_update_threshold",
        config.procedural_memory.evolution_threshold,
    )
    session.update_session(
        "procedural_memory_best_of_n", config.procedural_memory.best_of_n
    )
    session.update_session(
        "procedural_memory_ppo_epsilon", config.procedural_memory.ppo_epsilon
    )
    session.update_session(
        "procedural_memory_selection_epsilon",
        config.procedural_memory.selection_epsilon,
    )
    session.update_session(
        "procedural_memory_experience_pool_size",
        config.procedural_memory.experience_pool_size,
    )
    session.update_session(
        "procedural_memory_baseline_ema_alpha",
        config.procedural_memory.baseline_ema_alpha,
    )
    session.update_session(
        "procedural_memory_selection_epsilon_decay_cases",
        config.procedural_memory.selection_epsilon_decay_cases,
    )
    session.update_session(
        "procedural_memory_acceptance_margin",
        config.procedural_memory.acceptance_margin,
    )
    session.update_session(
        "procedural_memory_verifier", config.procedural_memory.verifier
    )
    session.update_session(
        "procedural_memory_holdout_size", config.procedural_memory.holdout_size
    )
    session.update_session(
        "procedural_memory_min_positive_advantage",
        config.procedural_memory.min_positive_advantage,
    )
    session.update_session(
        "procedural_memory_evolver_model",
        config.procedural_memory.evolver_model,
    )
    session.update_session(
        "procedural_memory_policy_scorer_model",
        config.procedural_memory.policy_scorer_model,
    )
    session.update_session("tool_refinement_enabled", config.tool_refinement.enabled)
    session.update_session(
        "tool_refinement_state_path", str(config.tool_refinement.state_path or "")
    )
    session.update_session(
        "tool_refinement_update_due", config.tool_refinement.update_due
    )
    session.update_session("tool_library_id", config.tool_refinement.library_id)
    session.update_session("tool_doc_chars", config.tool_refinement.tool_doc_chars)
    session.update_session(
        "tool_convergence_threshold",
        config.tool_refinement.convergence_threshold,
    )
    session.update_session(
        "tool_exploration_similarity_threshold",
        config.tool_refinement.exploration_similarity_threshold,
    )
    session.update_session(
        "tool_explorer_reflection_limit",
        config.tool_refinement.explorer_reflection_limit,
    )
    session.update_session(
        "tool_refinement_update_interval", config.tool_refinement.update_interval
    )
    session.update_session(
        "tool_refinement_min_new_trials", config.tool_refinement.min_new_trials
    )
    session.update_session(
        "tool_refinement_max_tools_per_update",
        config.tool_refinement.max_tools_per_update,
    )
    session.update_session(
        "tool_refinement_publish_min_utility",
        config.tool_refinement.publish_min_utility,
    )
    session.update_session("tool_explorer_model", config.tool_refinement.explorer_model)
    session.update_session("tool_analyzer_model", config.tool_refinement.analyzer_model)
    session.update_session("tool_rewriter_model", config.tool_refinement.rewriter_model)


def start_agent(config: AgentRunConfig, *, session_id: str | None = None) -> None:
    """Use upstream execution unchanged unless a learning module is enabled."""
    if config.normalized_agent_type == "react" and not config.extensions_enabled:
        start_nika_agent(
            agent_type="react",
            llm_provider=config.llm_provider,
            model=config.model,
            max_steps=config.max_steps,
            session_id=session_id,
        )
        return

    session = Session().load_running_session(session_id=session_id)
    config = AgentRunConfig(
        agent_type=config.agent_type,
        llm_provider=config.llm_provider,
        model=config.model,
        max_steps=config.max_steps,
        max_attempts=config.max_attempts,
        session_id=session.session_id,
        tool_refinement=config.tool_refinement,
        procedural_memory=config.procedural_memory,
        allow_learning_updates=config.allow_learning_updates,
    )
    validate_agent_composition(config)
    session.update_session("agent_type", config.normalized_agent_type)
    session.update_session("llm_provider", config.llm_provider)
    # Preserve the compatibility key consumed by existing learning artifacts.
    session.update_session("llm_backend", config.llm_provider)
    session.update_session("model", config.model)
    session.update_session("max_attempts", config.max_attempts)
    _write_extension_metadata(session, config)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting {config.normalized_agent_type} in session {session.session_id}",
        session_id=session.session_id,
        agent_type=config.normalized_agent_type,
        model=config.model,
        procedural_memory=config.procedural_memory.enabled,
        allow_learning_updates=config.allow_learning_updates,
        tool_refinement=config.tool_refinement.enabled,
    )
    with mcp_gateway_for_session(
        session.session_id,
        scenario_name=session.scenario_name,
        policy_mode="two_phase",
    ):
        agent = create_extension_agent(config)
        asyncio.run(agent.run(task_description=session.task_description))

    session.end_session()
    log_event(
        "agent_end",
        f"Agent run completed for session {session.session_id}",
        session_id=session.session_id,
        agent_type=config.normalized_agent_type,
    )
