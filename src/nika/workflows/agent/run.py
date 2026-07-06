"""Run a troubleshooting agent against the current session task."""

import asyncio
import logging

from agent.registry import create_agent
from nika.utils.agent_config import (
    resolve_agent_model,
    resolve_agent_type,
    resolve_llm_provider,
    resolve_max_steps,
    resolve_reasoning_effort,
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
    reasoning_effort: str | None = None,
    stream_output: bool = True,
) -> None:
    """Load the running session, run the agent on ``task_description``, then end the session."""
    agent_type = resolve_agent_type(agent_type)
    max_steps = resolve_max_steps(max_steps)
    reasoning_effort = resolve_reasoning_effort(reasoning_effort)
    model = resolve_agent_model(agent_type, model)
    llm_provider = resolve_llm_provider(llm_provider, agent_type=agent_type)

    session = Session()
    session.load_running_session(session_id=session_id)
    session.update_session("agent_type", agent_type)
    if llm_provider is not None:
        session.update_session("llm_provider", llm_provider)
    session.update_session("model", model)
    if reasoning_effort is not None:
        session.update_session("reasoning_effort", reasoning_effort)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting agent: {agent_type} (model={model}) in session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_type,
        model=model,
    )
    if agent_type == "local_cli.codex_cli" and stream_output:
        effort_line = f" | Reasoning effort: {reasoning_effort}" if reasoning_effort else ""
        print(
            f"Session {session.session_id}\n"
            f"Agent: local_cli.codex_cli | Model: {model}{effort_line}\n"
            f"Results: {session.session_dir}\n",
            flush=True,
        )
    agent = create_agent(
        agent_type,
        session_id=session.session_id,
        llm_provider=llm_provider,
        model=model,
        max_steps=max_steps,
        reasoning_effort=reasoning_effort,
        stream_output=stream_output,
    )
    try:
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
    if agent_type == "local_cli.codex_cli" and stream_output:
        print(f"\nDone. Results saved to {session.session_dir}\n", flush=True)
