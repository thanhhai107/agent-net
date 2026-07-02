"""LangGraph + Claude Code CLI agent.

Reuses the same :class:`~langgraph.graph.StateGraph` structure as
:class:`~agent.byo.langgraph.react_agent.BasicReActAgent` but replaces the
LangChain ReAct workers with Claude Code CLI subprocess wrappers:

* **diagnosis phase** → :class:`~agent.local_cli.claude_cli.phases.ClaudeDiagnosisPhase`
  (``claude -p`` with Kathara MCP servers; server set chosen dynamically
  based on the session scenario)
* **submission phase** → :class:`~agent.local_cli.claude_cli.phases.ClaudeSubmissionPhase`
  (``claude -p`` with the task MCP server; calls ``submit()`` to record
  a structured result)

Authentication supports environment API keys, third-party Anthropic-compatible
endpoints (``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``), and
``claude auth login``.  See :mod:`agent.local_cli.claude_cli.config` and
``src/agent/README.md``.

Select with ``nika agent run -a local_cli.claude_cli``.
"""

import logging
import os
import sys
from typing import Any

import langsmith as ls
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import Field
from typing_extensions import TypedDict

from agent.local_cli.claude_cli.config import resolve_claude_model
from agent.local_cli.claude_cli.phases.diagnosis import ClaudeDiagnosisPhase
from agent.local_cli.claude_cli.phases.submission import ClaudeSubmissionPhase
from agent.utils.loggers import MessageLogger
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session import Session

logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: list[BaseMessage]
    diagnosis_report: str = Field(default="")
    is_max_steps_reached: bool = Field(default=False)


class ClaudeAgent:
    """Two-phase troubleshooting agent: LangGraph orchestration + Claude Code CLI workers.

    Parameters
    ----------
    session_id:
        NIKA session identifier.
    model:
        Claude model name forwarded to ``claude --model``.  When omitted,
        reads from environment (see :func:`~agent.local_cli.claude_cli.config.default_claude_model`).
    """

    def __init__(
        self,
        session_id: str,
        model: str | None = None,
        *,
        stream_output: bool = True,
    ) -> None:
        self.session_id = session_id
        self.model = resolve_claude_model(model)
        self._stream_output = stream_output

        session = Session()
        session.load_running_session(session_id=session_id)
        self.session = session
        self.session_dir: str = session.session_dir

        scenario_name: str = getattr(session, "scenario_name", "")
        problem_names: list[str] = getattr(session, "problem_names", [])

        self._diagnosis_phase = ClaudeDiagnosisPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=self.model,
            scenario_name=scenario_name,
            problem_names=problem_names,
            stream_output=stream_output,
        )
        self._submission_phase = ClaudeSubmissionPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=self.model,
            stream_output=stream_output,
        )

        builder = StateGraph(AgentState)
        builder.add_node(DIAGNOSIS, self._run_diagnosis)
        builder.add_node(SUBMISSION, self._run_submission)
        builder.add_edge(START, DIAGNOSIS)
        builder.add_conditional_edges(
            DIAGNOSIS,
            lambda state: state.get("is_max_steps_reached", False),
            {True: END, False: SUBMISSION},
        )
        builder.add_edge(SUBMISSION, END)
        self.graph = builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, task_description: str) -> dict[str, Any]:
        """Execute the two-phase pipeline and return the final graph state."""
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": getattr(self.session, "scenario_name", ""),
                "problem": getattr(self.session, "problem_names", [""])[0],
                "topo_size": getattr(self.session, "scenario_topo_size", ""),
                "model": self.model,
                "agent": "local_cli.claude_cli",
            },
        ):
            return await self.graph.ainvoke(
                {"messages": [HumanMessage(content=task_description)]}
            )

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    async def _run_diagnosis(self, state: AgentState) -> dict[str, Any]:
        task_description: str = state["messages"][-1].content
        logger = MessageLogger(agent=DIAGNOSIS, session_dir=self.session_dir)
        self._print_phase(DIAGNOSIS, "starting network fault analysis")
        logger.log("agent_start", {"phase": DIAGNOSIS, "task_preview": task_description[:200]})

        try:
            report = await self._diagnosis_phase.run(task_description)
        except Exception as exc:
            logger.log("agent_error", {"phase": DIAGNOSIS, "error": str(exc)})
            return {
                "diagnosis_report": f"ERROR: {exc}",
                "is_max_steps_reached": False,
            }

        is_error = report.startswith("ERROR:")
        logger.log("agent_done", {"phase": DIAGNOSIS, "is_error": is_error, "report_length": len(report)})
        self._print_phase(DIAGNOSIS, "completed" if not is_error else f"finished with error ({report[:120]})")
        return {
            "diagnosis_report": report,
            "is_max_steps_reached": False,
        }

    async def _run_submission(self, state: AgentState) -> dict[str, Any]:
        diagnosis_report: str = state["diagnosis_report"]
        logger = MessageLogger(agent=SUBMISSION, session_dir=self.session_dir)
        self._print_phase(SUBMISSION, "recording structured result")
        logger.log("agent_start", {"phase": SUBMISSION})

        try:
            result = await self._submission_phase.run(diagnosis_report)
        except Exception as exc:
            logger.log("agent_error", {"phase": SUBMISSION, "error": str(exc)})
            return {"messages": state["messages"]}

        logger.log("agent_done", {"phase": SUBMISSION, "result_length": len(result)})
        self._print_phase(SUBMISSION, "completed")
        return {"messages": [*state["messages"], HumanMessage(content=result)]}

    def _print_phase(self, phase: str, message: str) -> None:
        if not self._stream_output:
            return
        banner = f" [{phase.upper()}] {message} "
        width = max(60, len(banner) + 4)
        print(f"\n{'=' * width}", file=sys.stderr, flush=True)
        print(banner.center(width), file=sys.stderr, flush=True)
        print(f"{'=' * width}\n", file=sys.stderr, flush=True)
