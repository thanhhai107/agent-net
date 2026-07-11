"""Execution boundary for optional learning extensions."""

from __future__ import annotations

import asyncio

from agent.composition import AgentRunConfig, validate_agent_composition
from agent.extensions.react_agent import create_react_agent
from nika.service.mcp_gateway.lifecycle import mcp_gateway_for_session
from nika.utils.logger import bind_session_dir, log_event
from nika.utils.session import Session
from nika.workflows.agent.run import start_agent as start_nika_agent


def _write_extension_metadata(session: Session, config: AgentRunConfig) -> None:
    session.update_session("memory_mode", config.memory.mode)
    session.update_session("memory_bank", config.memory.bank)
    session.update_session("memory_top_k", config.memory.top_k)
    session.update_session("memory_token_budget", config.memory.token_budget)
    session.update_session("memory_max_skill_age", config.memory.max_skill_age)
    session.update_session("memory_pool_size", config.memory.pool_size)
    session.update_session(
        "memory_evolution_threshold", config.memory.evolution_threshold
    )
    session.update_session("memory_best_of_n", config.memory.best_of_n)
    session.update_session("memory_ppo_epsilon", config.memory.ppo_epsilon)
    session.update_session(
        "tool_evolution_enabled", config.tool_evolution.enabled
    )
    session.update_session("tool_library_id", config.tool_evolution.library_id)
    session.update_session("tool_doc_chars", config.tool_evolution.tool_doc_chars)
    session.update_session(
        "tool_convergence_threshold",
        config.tool_evolution.convergence_threshold,
    )


def start_agent(config: AgentRunConfig, *, session_id: str | None = None) -> None:
    """Use upstream execution unchanged unless a learning module is enabled."""
    if not config.extensions_enabled:
        start_nika_agent(
            agent_type="byo.langgraph",
            llm_provider=config.llm_provider,
            model=config.model,
            max_steps=config.max_steps,
            session_id=session_id,
            stream_output=False,
        )
        return

    session = Session().load_running_session(session_id=session_id)
    config = AgentRunConfig(
        agent_type=config.agent_type,
        llm_provider=config.llm_provider,
        model=config.model,
        max_steps=config.max_steps,
        session_id=session.session_id,
        tool_evolution=config.tool_evolution,
        memory=config.memory,
    )
    validate_agent_composition(config)
    session.update_session("agent_type", "byo.langgraph")
    session.update_session("llm_provider", config.llm_provider)
    # Preserve the compatibility key consumed by existing learning artifacts.
    session.update_session("llm_backend", config.llm_provider)
    session.update_session("model", config.model)
    _write_extension_metadata(session, config)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting NIKA ReAct with learning extensions in session {session.session_id}",
        session_id=session.session_id,
        agent_type="byo.langgraph",
        model=config.model,
        memory=config.memory.mode,
        tool_evolution=config.tool_evolution.enabled,
    )
    with mcp_gateway_for_session(
        session.session_id,
        scenario_name=session.scenario_name,
        policy_mode="two_phase",
    ):
        agent = create_react_agent(config)
        asyncio.run(agent.run(task_description=session.task_description))

    session.end_session()
    log_event(
        "agent_end",
        f"Agent run completed for session {session.session_id}",
        session_id=session.session_id,
        agent_type="byo.langgraph",
    )

