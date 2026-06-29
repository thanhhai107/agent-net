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


def start_agent(agent_config: AgentRunConfig, *, session_id: str | None = None) -> None:
    """Load the running session, run the agent on ``task_description``, then end the session."""
    validate_agent_extensions(agent_config)
    session = Session()
    session.load_running_session(session_id=session_id)
    agent_config = replace(agent_config, session_id=session.session_id)
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
    if agent_config.reasoning_effort is not None:
        session.update_session("reasoning_effort", agent_config.reasoning_effort)
    if agent_config.policy_overlay.enabled:
        session.update_session("policy_overlay_path", agent_config.policy_overlay.path)
    session.update_session("oracle_routing", agent_config.oracle_routing)
    session.update_session("tool_evolution_enabled", agent_config.tool_evolution.enabled)
    if agent_config.tool_evolution.enabled:
        session.update_session("tool_library_id", agent_config.tool_evolution.library_id)
        session.update_session("tool_evolution_mode", agent_config.tool_evolution.mode)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting agent: {agent_config.agent_type} (model={agent_config.model}) in session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_config.agent_type,
        model=agent_config.model,
    )
    if agent_config.normalized_agent_type == "cli" and agent_config.stream_output:
        effort_line = (
            f" | Reasoning effort: {agent_config.reasoning_effort}"
            if agent_config.reasoning_effort
            else ""
        )
        print(
            f"Session {session.session_id}\n"
            f"Agent: cli | Model: {agent_config.model}{effort_line}\n"
            f"Results: {session.session_dir}\n",
            flush=True,
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
    if agent_config.normalized_agent_type == "cli" and agent_config.stream_output:
        print(f"\nDone. Results saved to {session.session_dir}\n", flush=True)
