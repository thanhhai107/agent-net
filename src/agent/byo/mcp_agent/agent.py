"""LangGraph + mcp-agent SDK agent.

Two-phase troubleshooting pipeline:

* **diagnosis phase** → :class:`~agent.byo.mcp_agent.phases.McpDiagnosisPhase`
  (mcp-agent Agent with Kathara MCP servers)
* **submission phase** → :class:`~agent.byo.mcp_agent.phases.McpSubmissionPhase`
  (mcp-agent Agent with the task MCP server)

Select with ``nika agent run -a byo.mcp_agent``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from mcp_agent.app import MCPApp
from pydantic import Field
from typing_extensions import TypedDict

from agent.byo.mcp_agent.config import build_mcp_agent_settings, diagnosis_server_names
from agent.byo.mcp_agent.phases.diagnosis import McpDiagnosisPhase
from agent.byo.mcp_agent.phases.submission import McpSubmissionPhase
from agent.utils.loggers import MessageLogger
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session import Session

logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: list[BaseMessage]
    diagnosis_report: str = Field(default="")
    is_max_steps_reached: bool = Field(default=False)


class McpAgent:
    """Two-phase troubleshooting agent: LangGraph orchestration + mcp-agent workers."""

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4.1-mini",
        max_steps: int = 20,
        *,
        stream_output: bool = True,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.max_steps = max_steps
        self._stream_output = stream_output

        session = Session()
        session.load_running_session(session_id=session_id)
        self.session = session
        self.session_dir: str = session.session_dir

        self._scenario_name: str = getattr(session, "scenario_name", "")
        self._problem_names: list[str] = getattr(session, "problem_names", [])
        self._diagnosis_server_names = diagnosis_server_names(
            self._scenario_name, self._problem_names
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

    async def run(self, task_description: str) -> dict[str, Any]:
        """Execute the two-phase pipeline inside an MCPApp context."""
        settings = build_mcp_agent_settings(
            session_id=self.session_id,
            scenario_name=self._scenario_name,
            problem_names=self._problem_names,
            model=self.model,
        )
        app = MCPApp(name="nika_mcp_agent", settings=settings, session_id=self.session_id)
        async with app.run():
            return await self.graph.ainvoke(
                {"messages": [HumanMessage(content=task_description)]}
            )

    async def _run_diagnosis(self, state: AgentState) -> dict[str, Any]:
        task_description: str = state["messages"][-1].content
        logger = MessageLogger(agent=DIAGNOSIS, session_dir=self.session_dir)
        self._print_phase(DIAGNOSIS, "starting network fault analysis")
        logger.log("agent_start", {"phase": DIAGNOSIS, "task_preview": task_description[:200]})

        phase = McpDiagnosisPhase(
            session_dir=self.session_dir,
            model=self.model,
            max_steps=self.max_steps,
            server_names=self._diagnosis_server_names,
        )
        try:
            report, is_max_steps_reached = await phase.run(task_description)
        except Exception as exc:
            logger.log("agent_error", {"phase": DIAGNOSIS, "error": str(exc)})
            return {
                "diagnosis_report": f"ERROR: {exc}",
                "is_max_steps_reached": False,
            }

        if is_max_steps_reached:
            logger.log("error", {"message": "Diagnosis phase reached max iteration limit."})
            logger.log("agent_done", {"phase": DIAGNOSIS, "is_error": True, "report_length": len(report)})
            self._print_phase(DIAGNOSIS, "stopped: max steps reached")
            return {
                "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                "is_max_steps_reached": True,
            }

        is_error = report.startswith("ERROR:")
        logger.log("agent_done", {"phase": DIAGNOSIS, "is_error": is_error, "report_length": len(report)})
        self._print_phase(
            DIAGNOSIS,
            "completed" if not is_error else f"finished with error ({report[:120]})",
        )
        return {
            "diagnosis_report": report,
            "is_max_steps_reached": False,
        }

    async def _run_submission(self, state: AgentState) -> dict[str, Any]:
        diagnosis_report: str = state["diagnosis_report"]
        logger = MessageLogger(agent=SUBMISSION, session_dir=self.session_dir)
        self._print_phase(SUBMISSION, "recording structured result")
        logger.log("agent_start", {"phase": SUBMISSION})

        phase = McpSubmissionPhase(
            session_dir=self.session_dir,
            model=self.model,
            max_steps=self.max_steps,
        )
        try:
            result = await phase.run(diagnosis_report)
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
