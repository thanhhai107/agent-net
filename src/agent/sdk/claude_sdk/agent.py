"""Claude Agent SDK troubleshooting agent.

Two-phase pipeline via native ``ClaudeSDKClient`` sessions (no LangGraph).
Select with ``nika agent run -a sdk.claude_sdk``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import langsmith as ls

from agent.sdk.claude_sdk.config import resolve_claude_sdk_model
from agent.sdk.claude_sdk.phases.diagnosis import ClaudeSdkDiagnosisPhase
from agent.sdk.claude_sdk.phases.submission import ClaudeSdkSubmissionPhase
from agent.utils.phases import DIAGNOSIS, SUBMISSION
from nika.utils.session import Session


class ClaudeSdkAgent:
    """Two-phase troubleshooting agent backed by claude-agent-sdk."""

    def __init__(
        self,
        session_id: str,
        model: str | None = None,
        max_steps: int = 20,
        *,
        stream_output: bool = True,
    ) -> None:
        self.session_id = session_id
        self.model = resolve_claude_sdk_model(model)
        self.max_steps = max_steps
        self._stream_output = stream_output

        session = Session()
        session.load_running_session(session_id=session_id)
        self.session = session
        self.session_dir: str = session.session_dir

        scenario_name: str = getattr(session, "scenario_name", "")

        self._diagnosis_phase = ClaudeSdkDiagnosisPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=self.model,
            max_steps=max_steps,
            scenario_name=scenario_name,
        )
        self._submission_phase = ClaudeSdkSubmissionPhase(
            session_id=session_id,
            session_dir=self.session_dir,
            model=self.model,
            max_steps=max_steps,
        )

    async def run(self, task_description: str) -> dict[str, Any]:
        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": getattr(self.session, "scenario_name", ""),
                "problem": getattr(self.session, "problem_names", [""])[0],
                "topo_size": getattr(self.session, "scenario_topo_size", ""),
                "model": self.model,
                "agent": "sdk.claude_sdk",
            },
        ):
            self._print_phase(DIAGNOSIS, "starting network fault analysis")
            diagnosis_report = await self._diagnosis_phase.run(task_description)
            self._print_phase(
                DIAGNOSIS,
                "completed"
                if not diagnosis_report.startswith("ERROR:")
                else diagnosis_report[:120],
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
