"""OpenAI Codex SDK troubleshooting agent.

Two-phase pipeline via native ``AsyncCodex`` threads (no LangGraph).
Select with ``nika agent run -a sdk.codex_sdk``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import langsmith as ls

from agent.sdk.codex_sdk.phases.diagnosis import CodexSdkDiagnosisPhase
from agent.sdk.codex_sdk.phases.submission import CodexSdkSubmissionPhase
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session import Session


class CodexSdkAgent:
    """Two-phase troubleshooting agent backed by openai-codex."""

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str | None = None,
        *,
        stream_output: bool = True,
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._stream_output = stream_output

        session = Session()
        session.load_running_session(session_id=session_id)
        self.session = session
        self.session_dir: str = session.session_dir

        scenario_name: str = getattr(session, "scenario_name", "")
        problem_names: list[str] = getattr(session, "problem_names", [])

        self._diagnosis_phase = CodexSdkDiagnosisPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            scenario_name=scenario_name,
            problem_names=problem_names,
            stream_output=stream_output,
        )
        self._submission_phase = CodexSdkSubmissionPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            stream_output=stream_output,
        )

    async def run(self, task_description: str) -> dict[str, Any]:
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": getattr(self.session, "scenario_name", ""),
                "problem": getattr(self.session, "problem_names", [""])[0],
                "topo_size": getattr(self.session, "scenario_topo_size", ""),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "agent": "sdk.codex_sdk",
            },
        ):
            self._print_phase(DIAGNOSIS, "starting network fault analysis")
            diagnosis_report = await self._diagnosis_phase.run(task_description)
            self._print_phase(
                DIAGNOSIS,
                "completed" if not diagnosis_report.startswith("ERROR:") else diagnosis_report[:120],
            )

            self._print_phase(SUBMISSION, "recording structured result")
            submission_result = await self._submission_phase.run(diagnosis_report)
            self._print_phase(SUBMISSION, "completed")

            return {
                "diagnosis_report": diagnosis_report,
                "submission_result": submission_result,
            }

    def _print_phase(self, phase: str, message: str) -> None:
        if not self._stream_output:
            return
        banner = f" [{phase.upper()}] {message} "
        width = max(60, len(banner) + 4)
        print(f"\n{'=' * width}", file=sys.stderr, flush=True)
        print(banner.center(width), file=sys.stderr, flush=True)
        print(f"{'=' * width}\n", file=sys.stderr, flush=True)
