"""Run a troubleshooting agent against the current session task."""

import asyncio
import logging

from agent.registry import MEMORY_COMPATIBLE_AGENT_TYPES, create_agent
from nika.utils.logger import bind_session_dir, log_event
from nika.utils.session import Session

logging.basicConfig(level=logging.INFO)


def start_agent(
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    *,
    max_attempts: int = 3,
    session_id: str | None = None,
    reasoning_effort: str | None = None,
    stream_output: bool = True,
    oracle_routing: bool = False,
    tool_evolution_enabled: bool = False,
    tool_library_id: str = "default",
    tool_evolution_mode: str = "dual",
    memory_mode: str = "off",
    memory_bank: str = "default",
    memory_top_k: int = 5,
    memory_token_budget: int = 1500,
) -> None:
    """Load the running session, run the agent on ``task_description``, then end the session."""
    if memory_mode not in {"off", "read", "evolve"}:
        raise ValueError("memory_mode must be one of: off, read, evolve")
    normalized_type = agent_type.lower()
    if tool_evolution_enabled and normalized_type not in MEMORY_COMPATIBLE_AGENT_TYPES:
        raise ValueError(
            "Tool Evolution supports react, plan-execute, and reflexion workflows."
        )
    if memory_mode != "off" and normalized_type not in MEMORY_COMPATIBLE_AGENT_TYPES:
        supported = ", ".join(sorted(MEMORY_COMPATIBLE_AGENT_TYPES))
        raise ValueError(f"memory is supported only for these workflows: {supported}")

    session = Session()
    session.load_running_session(session_id=session_id)
    session.update_session("agent_type", agent_type)
    session.update_session("llm_backend", llm_backend)
    session.update_session("model", model)
    if agent_type == "reflexion":
        session.update_session("max_attempts", max_attempts)
    session.update_session("memory_mode", memory_mode)
    if memory_mode != "off":
        session.update_session("memory_bank", memory_bank)
        session.update_session("memory_top_k", memory_top_k)
        session.update_session("memory_token_budget", memory_token_budget)
    if reasoning_effort is not None:
        session.update_session("reasoning_effort", reasoning_effort)
    session.update_session("oracle_routing", oracle_routing)
    session.update_session("tool_evolution_enabled", tool_evolution_enabled)
    if tool_evolution_enabled:
        session.update_session("tool_library_id", tool_library_id)
        session.update_session("tool_evolution_mode", tool_evolution_mode)
    session.start_session()

    bind_session_dir(session.session_dir)
    log_event(
        "agent_start",
        f"Starting agent: {agent_type} (model={model}) in session {session.session_id}",
        session_id=session.session_id,
        agent_type=agent_type,
        model=model,
    )
    if agent_type == "cli" and stream_output:
        effort_line = (
            f" | Reasoning effort: {reasoning_effort}" if reasoning_effort else ""
        )
        print(
            f"Session {session.session_id}\n"
            f"Agent: cli | Model: {model}{effort_line}\n"
            f"Results: {session.session_dir}\n",
            flush=True,
        )
    try:
        agent = create_agent(
            agent_type,
            session_id=session.session_id,
            llm_backend=llm_backend,
            model=model,
            max_steps=max_steps,
            max_attempts=max_attempts,
            reasoning_effort=reasoning_effort,
            stream_output=stream_output,
            oracle_routing=oracle_routing,
            tool_evolution_enabled=tool_evolution_enabled,
            tool_library_id=tool_library_id,
            tool_evolution_mode=tool_evolution_mode,
            memory_mode=memory_mode,
            memory_bank=memory_bank,
            memory_top_k=memory_top_k,
            memory_token_budget=memory_token_budget,
        )
        asyncio.run(agent.run(task_description=session.task_description))
    finally:
        session.end_session()
        log_event(
            "agent_end",
            f"Agent run ended for session {session.session_id}",
            session_id=session.session_id,
            agent_type=agent_type,
        )
    if agent_type == "cli" and stream_output:
        print(f"\nDone. Results saved to {session.session_dir}\n", flush=True)
