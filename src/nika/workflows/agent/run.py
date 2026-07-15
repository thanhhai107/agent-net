"""Run a troubleshooting agent against the current session task."""

import asyncio
import logging

from agent.registry import create_agent
from nika.service.mcp_gateway.lifecycle import mcp_gateway_for_session
from nika.utils.agent_config import (
    resolve_agent_model,
    resolve_agent_type,
    resolve_llm_provider,
    resolve_max_steps,
)
from nika.utils.logger import bind_session_dir, log_error_event, log_event
from nika.utils.session import Session

logging.basicConfig(level=logging.INFO)


def start_agent(
    agent_type: str | None = None,
    llm_provider: str | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    *,
    session_id: str | None = None,
) -> None:
    """Load the running session, run the agent on ``task_description``, then end the session."""
    agent_type = resolve_agent_type(agent_type)
    max_steps = resolve_max_steps(max_steps)
    model = resolve_agent_model(agent_type, model)
    llm_provider = resolve_llm_provider(llm_provider, agent_type=agent_type)

    session = Session()
    session.load_running_session(session_id=session_id)
    session.update_session("agent_type", agent_type)
    if llm_provider is not None:
        session.update_session("llm_provider", llm_provider)
    session.update_session("model", model)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting agent: {agent_type} (model={model}) in session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_type,
        model=model,
    )
    try:
        with mcp_gateway_for_session(
            session.session_id,
            scenario_name=session.scenario_name,
            policy_mode="two_phase",
        ):
            agent = create_agent(
                agent_type,
                session_id=session.session_id,
                llm_provider=llm_provider,
                model=model,
                max_steps=max_steps,
            )
            asyncio.run(agent.run(task_description=session.task_description))
    except Exception as exc:
        log_error_event(
            "agent_error",
            f"Agent run failed for session {session.session_id}: {exc}",
            session_id=session.session_id,
            agent_type=agent_type,
            model=model,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise

    session.end_session()
    log_event(
        "agent_end",
        f"Agent run completed for session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_type,
    )
