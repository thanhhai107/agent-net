"""Run a troubleshooting agent against the current session task."""

import asyncio
import logging
from dataclasses import replace

from agent.composition import (
    AgentRunConfig,
    validate_agent_extensions,
    validate_agent_composition,
)
from agent.registry import create_agent
from nika.utils.logger import bind_session_dir, log_event
from nika.utils.session import Session

logging.basicConfig(level=logging.INFO)


def _resolve_requested_model(
    agent_config: AgentRunConfig,
    *,
    requested_model: str | None,
) -> str:
    if requested_model:
        return requested_model
    return agent_config.model


def start_agent(
    agent_config: AgentRunConfig,
    *,
    session_id: str | None = None,
    requested_model: str | None = None,
) -> None:
    """Load the running session, run the agent on ``task_description``, then end the session."""
    validate_agent_extensions(agent_config)
    session = Session()
    session.load_running_session(session_id=session_id)
    agent_config = replace(
        agent_config,
        session_id=session.session_id,
        model=_resolve_requested_model(
            agent_config,
            requested_model=requested_model,
        ),
    )
    validate_agent_composition(agent_config)

    session.update_session("agent_type", agent_config.agent_type)
    session.update_session("llm_backend", agent_config.llm_backend)
    session.update_session("model", agent_config.model)
    if agent_config.normalized_agent_type == "reflexion":
        session.update_session("max_attempts", agent_config.max_attempts)
    session.update_session("memory_mode", agent_config.memory.mode)
    if agent_config.memory.enabled:
        session.update_session("memory_bank", agent_config.memory.bank)
        session.update_session("memory_top_k", agent_config.memory.top_k)
        session.update_session("memory_token_budget", agent_config.memory.token_budget)
        session.update_session(
            "memory_skill_selector_mode",
            agent_config.memory.skill_selector_mode,
        )
        session.update_session(
            "memory_meta_controller_mode",
            agent_config.memory.meta_controller_mode,
        )
        session.update_session("memory_max_skill_age", agent_config.memory.max_skill_age)
        session.update_session(
            "memory_selector_min_lcb",
            agent_config.memory.selector_min_lcb,
        )
        session.update_session(
            "memory_selector_nominee_k",
            agent_config.memory.selector_nominee_k,
        )
        session.update_session("memory_pool_size", agent_config.memory.pool_size)
        session.update_session(
            "memory_evolution_threshold",
            agent_config.memory.evolution_threshold,
        )
        session.update_session("memory_best_of_n", agent_config.memory.best_of_n)
        session.update_session("memory_ppo_epsilon", agent_config.memory.ppo_epsilon)
    session.update_session("tool_evolution_enabled", agent_config.tool_evolution.enabled)
    if agent_config.tool_evolution.enabled:
        session.update_session("tool_library_id", agent_config.tool_evolution.library_id)
        session.update_session("tool_doc_chars", agent_config.tool_evolution.tool_doc_chars)
        session.update_session(
            "tool_prompt_doc_limit",
            agent_config.tool_evolution.prompt_doc_limit,
        )
        session.update_session(
            "tool_scoped_prompt_doc_limit",
            agent_config.tool_evolution.scoped_prompt_doc_limit,
        )
        session.update_session(
            "tool_planned_checks",
            agent_config.tool_evolution.planned_checks,
        )
        session.update_session("tool_next_checks", agent_config.tool_evolution.next_checks)
        session.update_session(
            "tool_convergence_threshold",
            agent_config.tool_evolution.convergence_threshold,
        )
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting agent: {agent_config.agent_type} (model={agent_config.model}) in session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_config.agent_type,
        model=agent_config.model,
    )
    try:
        agent = create_agent(agent_config)
        asyncio.run(agent.run(task_description=session.task_description))
    finally:
        session.end_session()
        log_event(
            "agent_end",
            f"Agent run ended for session {session.session_id}",
            session_id=session.session_id,
            agent_type=agent_config.agent_type,
        )
