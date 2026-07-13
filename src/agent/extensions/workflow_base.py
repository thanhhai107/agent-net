"""Shared lifecycle for local workflows built on NIKA's original phases."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from agent.byo.langgraph.phases.submission import SubmissionPhase
from agent.composition import AgentRunConfig
from agent.extensions.react_agent import (
    LearningDiagnosisPhase,
    configure_custom_provider_environment,
)
from agent.tool_refinement.integration import write_tool_refinement_session
from agent.utils.loggers import AgentCallbackLogger, MessageLogger
from agent.utils.mcp_client import begin_submission_mcp_phase
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session import Session


class _PhaseMessageLogger(MessageLogger):
    def __init__(self, *, agent: str, session_dir: str, phase: str) -> None:
        super().__init__(agent=agent, session_dir=session_dir)
        self.phase = phase

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        super().log(event_type, {"phase": self.phase, **payload})


class _PhaseCallbackLogger(AgentCallbackLogger):
    def __init__(self, *, session_dir: str, phase: str) -> None:
        super().__init__(agent=DIAGNOSIS, session_dir=session_dir)
        self._logger = _PhaseMessageLogger(
            agent=DIAGNOSIS,
            session_dir=session_dir,
            phase=phase,
        )


class ExtensionWorkflowBase:
    """Load the original NIKA tools and preserve extension artifacts."""

    workflow_name = "extension"

    def __init__(self, config: AgentRunConfig) -> None:
        configure_custom_provider_environment()
        self.config = config
        self.max_steps = config.max_steps
        self.session = Session().load_running_session(session_id=config.session_id)
        self.session_id = self.session.session_id
        self.session_dir = self.session.session_dir
        self.diagnosis_phase = LearningDiagnosisPhase(config)
        asyncio.run(self.diagnosis_phase.load_tools())
        self.llm = self.diagnosis_phase.llm

    def prepare_diagnosis(self, task_description: str):
        if self.config.procedural_memory.enabled:
            self.diagnosis_phase.install_procedural_memory(
                task_description, self.session_dir
            )
        return self.diagnosis_phase.get_agent()

    def callback(self, phase: str) -> AgentCallbackLogger:
        return _PhaseCallbackLogger(
            session_dir=self.session_dir,
            phase=phase,
        )

    def log_error(self, phase: str, exc: Exception | str) -> None:
        _PhaseMessageLogger(
            agent=DIAGNOSIS,
            session_dir=self.session_dir,
            phase=phase,
        ).log("error", {"message": str(exc)})

    async def submit(self, diagnosis_report: str) -> dict[str, Any]:
        if not diagnosis_report.strip():
            return {}
        begin_submission_mcp_phase(self.session_id)
        phase = SubmissionPhase(
            session_id=self.session_id,
            llm_provider=self.config.llm_provider,
            model=self.config.model,
            scenario_name=self.session.scenario_name,
        )
        await phase.load_tools()
        try:
            result = await phase.get_agent().ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Based on the diagnosis report: {diagnosis_report}, "
                                "please provide the submission. Do not submit if no "
                                "report available."
                            )
                        )
                    ]
                },
                config={
                    "callbacks": [
                        AgentCallbackLogger(
                            agent=SUBMISSION,
                            session_dir=self.session_dir,
                        )
                    ],
                    "recursion_limit": self.max_steps,
                },
            )
        except GraphRecursionError:
            return {
                "diagnosis_report": diagnosis_report,
                "is_max_steps_reached": True,
            }
        return {"messages": result["messages"], "diagnosis_report": diagnosis_report}

    async def explore_tools(self, task_description: str) -> dict[str, Any]:
        diagnosis_phase = getattr(self, "diagnosis_phase", None)
        runtime = getattr(diagnosis_phase, "tool_refinement_runtime", None)
        if runtime is None:
            return {}
        return await runtime.explore(task_description)

    def write_extension_snapshots(self) -> None:
        write_tool_refinement_session(
            self.diagnosis_phase.tool_refinement_runtime,
            self.session_dir,
        )
        runtime = self.diagnosis_phase.skill_tool_runtime
        if runtime is None:
            return
        (Path(self.session_dir) / "procedural_memory_runtime_session.json").write_text(
            json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
