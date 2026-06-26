"""Codex CLI-backed diagnosis phase worker.

Mirrors the role of :class:`~agent.langgraph.phases.DiagnosisPhase`
in the LangChain path: wraps the same troubleshooting prompt and exposes an
async ``run()`` that returns a free-text diagnosis report.
"""

from agent.codex_cli.codex_worker import CodexWorker
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from agent.utils.phases import DIAGNOSIS


class CodexCliDiagnosisPhase:
    """Runs network fault diagnosis via a ``codex exec`` subprocess.

    Parameters
    ----------
    session_id:
        NIKA session identifier.
    session_dir:
        Absolute path to the session results directory.
    model:
        Codex model name (e.g. ``"gpt-5.4-mini"``).
    reasoning_effort:
        Optional Codex ``model_reasoning_effort`` override.
    timeout:
        Hard timeout in seconds for the subprocess.
    scenario_name:
        Scenario identifier used to select relevant Kathara MCP servers.
    problem_names:
        Problem identifiers used together with *scenario_name* for server selection.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: str,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str | None = None,
        timeout: int = 600,
        scenario_name: str = "",
        *,
        stream_output: bool = True,
    ) -> None:
        self._worker = CodexWorker(
            session_id=session_id,
            session_dir=session_dir,
            phase=DIAGNOSIS,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            scenario_name=scenario_name,
            stream_output=stream_output,
        )

    async def run(self, task_description: str) -> str:
        """Return a free-text diagnosis report produced by ``codex exec``."""
        prompt = f"{OVERALL_DIAGNOSIS_PROMPT}\n\nTask: {task_description}"
        return await self._worker.run(prompt)
